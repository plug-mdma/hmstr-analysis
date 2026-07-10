"""
Количественное исследование: пампы HMSTR/USDT и возврат к среднему.

Что делает скрипт:
1. Скачивает исторические OHLCV-свечи HMSTR/USDT с биржи (по умолчанию Binance).
2. Находит эпизоды "пампа" — рост цены более чем на X% за окно T часов.
3. Для каждого эпизода считает: глубину и время последующей просадки,
   был ли повторный рост ("вторая волна") и через сколько времени.
4. Бэктестит идею "шорт после роста на X% до уровня N -> лонг на N с целью повтора роста".
5. Выводит статистику (не только среднее, но и медиану/квантили — на мемкоинах
   средние сильно искажены редкими экстремальными хвостами).

Установка зависимостей (один раз):
    pip install ccxt pandas numpy

Запуск:
    python hmstr_pump_reversion_research.py

Результаты:
    - hmstr_ohlcv.csv       — сырые свечи (кэшируются, чтобы не качать повторно)
    - pump_episodes.csv     — таблица найденных эпизодов пампа
    - backtest_trades.csv   — сделки по бэктесту шорт->лонг стратегии
    - в консоль печатается сводная статистика

НЮАНСЫ:
    - Если Binance заблокирован в твоём регионе, скрипт автоматически попробует Bybit/OKX/MEXC
    - Для разных бирж тикер может отличаться (HMSTR/USDT vs HMSTRUSDT vs HMSTR_USDT)
    - При ошибках RateLimit скрипт автоматически увеличивает паузу между запросами
"""

import ccxt
import pandas as pd
import numpy as np
import os
import time

# ============================================================
# 1. НАСТРОЙКИ — меняйте под свою гипотезу
# ============================================================

EXCHANGE_ID = "binance"       # можно "bybit", "okx", "gateio", "mexc", "bitget" и т.д.
SYMBOL = "HMSTR/USDT"
TIMEFRAME = "1h"              # "1m", "5m", "15m", "1h", "4h", "1d"
SINCE_DATE = "2024-09-01"     # HMSTR листился в сентябре 2024

PUMP_THRESHOLD_PCT = 30.0     # X% — минимальный рост за окно, чтобы считать это "пампом"
PUMP_WINDOW_HOURS = 24        # T — окно в часах, за которое ищем рост
COOLDOWN_HOURS = 48           # не считать новый эпизод пампа раньше, чем через столько часов

# Параметры для бэктеста стратегии "шорт после X% роста, лонг на уровне N"
SHORT_ENTRY_PCT = 30.0        # открыть шорт, когда рост от локального дна составил X%
SHORT_TARGET_RETRACE_PCT = 50.0   # закрыть шорт / открыть лонг, когда цена откатила на N% от размера самого пампа
LONG_TARGET_PCT = 30.0        # цель по лонгу — рост на столько же % от точки входа в лонг
STOP_LOSS_PCT = 15.0          # стоп по обеим ногам сделки, % от цены входа

DATA_FILE = "hmstr_ohlcv.csv"

# Список бирж для автоматического fallback (если основная недоступна)
FALLBACK_EXCHANGES = ["bybit", "okx", "mexc", "gateio", "bitget"]

# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================

def fetch_ohlcv_full(exchange_id, symbol, timeframe, since_date):
    """Загрузка OHLCV с автоматическим fallback на другие биржи при ошибках."""
    if os.path.exists(DATA_FILE):
        print(f"Найден кэш {DATA_FILE}, загружаю из него (удалите файл, чтобы перекачать заново).")
        df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        return df

    # Пробуем основную биржу и fallback-список
    exchanges_to_try = [exchange_id] + [ex for ex in FALLBACK_EXCHANGES if ex != exchange_id]
    exchange = None
    working_exchange_id = None
    
    for ex_id in exchanges_to_try:
        try:
            print(f"Пытаюсь подключиться к {ex_id}...")
            exchange_class = getattr(ccxt, ex_id)
            exchange = exchange_class({"enableRateLimit": True})
            
            # Проверяем доступность символа
            markets = exchange.load_markets()
            if symbol not in markets:
                print(f"  ⚠ Символ {symbol} не найден на {ex_id}")
                continue
            
            working_exchange_id = ex_id
            print(f"  ✓ Успешное подключение к {ex_id}")
            break
        except Exception as e:
            print(f"  ✗ Ошибка подключения к {ex_id}: {str(e)[:100]}")
            continue
    
    if exchange is None or working_exchange_id is None:
        raise RuntimeError(f"Не удалось подключиться ни к одной бирже из списка: {exchanges_to_try}")
    
    # Адаптируем символ под биржу если нужно
    actual_symbol = symbol
    try:
        # Пробуем загрузить одну свечу для проверки
        test = exchange.fetch_ohlcv(actual_symbol, timeframe=timeframe, limit=1)
    except Exception as e:
        # Если не работает, пробуем альтернативные форматы тикера
        alt_symbols = [
            symbol.replace("/", ""),           # HMSTRUSDT
            symbol.replace("/", "_"),          # HMSTR_USDT
            symbol + ":USDT",                  # HMSTR/USDT:USDT (фьючи)
        ]
        for alt in alt_symbols:
            try:
                test = exchange.fetch_ohlcv(alt, timeframe=timeframe, limit=1)
                actual_symbol = alt
                print(f"  ℹ Используем альтернативный формат тикера: {alt}")
                break
            except:
                continue
    
    since = exchange.parse8601(f"{since_date}T00:00:00Z")
    all_rows = []
    limit = 1000
    retry_count = 0
    max_retries = 5
    base_sleep = exchange.rateLimit / 1000

    print(f"Качаю {actual_symbol} {timeframe} с {working_exchange_id} начиная с {since_date}...")

    while True:
        try:
            candles = exchange.fetch_ohlcv(actual_symbol, timeframe=timeframe, since=since, limit=limit)
            if not candles:
                break
            all_rows.extend(candles)
            last_ts = candles[-1][0]
            if last_ts == since:
                break
            since = last_ts + 1
            
            # Адаптивная пауза с увеличением при RateLimit
            sleep_time = base_sleep * (1.5 ** retry_count)
            time.sleep(sleep_time)
            retry_count = 0  # сбрасываем при успехе
            
            if last_ts > exchange.milliseconds():
                break
                
        except ccxt.RateLimitExceeded as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"  ⚠ Превышен лимит попыток после RateLimit. Сохраняю то, что есть ({len(all_rows)} свечей)")
                break
            wait_time = min(60, base_sleep * (2 ** retry_count))
            print(f"  ⚠ RateLimit exceeded, жду {wait_time:.1f}с (попытка {retry_count}/{max_retries})")
            time.sleep(wait_time)
            continue
            
        except ccxt.NetworkError as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"  ⚠ Сетевая ошибка, сохраняю прогресс ({len(all_rows)} свечей)")
                break
            print(f"  ⚠ Сетевая ошибка: {str(e)[:80]}, повторяю через 5с")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Скачано {len(df)} свечей с {working_exchange_id}, сохранено в {DATA_FILE}")
    return df


# ============================================================
# 3. ПОИСК ЭПИЗОДОВ ПАМПА
# ============================================================

def find_pump_episodes(df, threshold_pct, window_hours, cooldown_hours, timeframe_hours):
    window_bars = max(1, int(window_hours / timeframe_hours))
    cooldown_bars = max(1, int(cooldown_hours / timeframe_hours))

    closes = df["close"].values
    highs = df["high"].values
    n = len(df)

    episodes = []
    last_episode_idx = -cooldown_bars

    for i in range(window_bars, n):
        if i - last_episode_idx < cooldown_bars:
            continue
        window_low = closes[i - window_bars:i].min()
        current_high = highs[i]
        if window_low <= 0:
            continue
        pct_move = (current_high - window_low) / window_low * 100

        if pct_move >= threshold_pct:
            local_low_idx = i - window_bars + np.argmin(closes[i - window_bars:i])
            episodes.append({
                "pump_start_idx": int(local_low_idx),
                "pump_peak_idx": int(i),
                "pump_start_time": df["timestamp"].iloc[local_low_idx],
                "pump_peak_time": df["timestamp"].iloc[i],
                "pump_start_price": float(closes[local_low_idx]),
                "pump_peak_price": float(current_high),
                "pump_pct_move": float(pct_move),
            })
            last_episode_idx = i

    return pd.DataFrame(episodes)


def analyze_reversion(df, episodes, lookahead_bars=200):
    """Для каждого пампа считаем максимальную последующую просадку и был ли повтор роста."""
    closes = df["close"].values
    n = len(df)

    results = []
    for _, ep in episodes.iterrows():
        peak_idx = ep["pump_peak_idx"]
        end_idx = min(n, peak_idx + lookahead_bars)
        if peak_idx + 1 >= end_idx:
            continue

        future = closes[peak_idx + 1:end_idx]
        peak_price = ep["pump_peak_price"]

        min_future = future.min()
        max_drawdown_pct = (peak_price - min_future) / peak_price * 100
        bars_to_min = int(np.argmin(future)) + 1

        # была ли "вторая волна" - рост от найденного дна более чем на 50% от размера
        # исходного пампа (в %), в пределах lookahead окна
        second_wave_pct = None
        second_wave_bars = None
        if bars_to_min < len(future):
            after_low = future[bars_to_min:]
            if len(after_low) > 0:
                rebound = (after_low.max() - min_future) / min_future * 100
                second_wave_pct = float(rebound)
                second_wave_bars = int(np.argmax(after_low))

        results.append({
            **ep.to_dict(),
            "max_drawdown_after_peak_pct": float(max_drawdown_pct),
            "bars_to_max_drawdown": bars_to_min,
            "second_wave_rebound_pct": second_wave_pct,
            "bars_to_second_wave": second_wave_bars,
        })

    return pd.DataFrame(results)


# ============================================================
# 4. БЭКТЕСТ: шорт после X% роста -> лонг на уровне N
# ============================================================

def backtest_short_then_long(df, short_entry_pct, retrace_target_pct,
                              long_target_pct, stop_loss_pct, window_hours, timeframe_hours):
    """
    Упрощённая логика на закрытии бара (без внутрибарового touch), без учёта
    комиссий/funding/проскальзывания -- добавьте их сами для реалистичности.

    Правила:
    - Ищем локальный минимум, от которого цена выросла на short_entry_pct% -> входим в шорт.
    - Шорт закрывается (переворот в лонг), когда цена откатила на retrace_target_pct%
      от размера самого движения (от пика вниз), либо когда сработал стоп-лосс.
    - Лонг закрывается по long_target_pct% от цены входа в лонг, либо по стоп-лоссу.
    """
    window_bars = max(1, int(window_hours / timeframe_hours))
    closes = df["close"].values
    times = df["timestamp"].values
    n = len(df)

    trades = []
    i = window_bars
    in_position = None  # None | dict с состоянием текущей связки шорт->лонг

    while i < n:
        price = closes[i]

        if in_position is None:
            window_low = closes[i - window_bars:i].min()
            if window_low > 0 and (price - window_low) / window_low * 100 >= short_entry_pct:
                in_position = {
                    "phase": "short",
                    "short_entry_idx": i,
                    "short_entry_price": price,
                    "pump_size_pct": (price - window_low) / window_low * 100,
                    "peak_price": price,
                }
            i += 1
            continue

        if in_position["phase"] == "short":
            in_position["peak_price"] = max(in_position["peak_price"], price)
            entry = in_position["short_entry_price"]

            # стоп-лосс по шорту (цена ушла выше)
            if (price - entry) / entry * 100 >= stop_loss_pct:
                trades.append({
                    "leg": "short", "entry_idx": in_position["short_entry_idx"],
                    "exit_idx": i, "entry_price": entry, "exit_price": price,
                    "pnl_pct": (entry - price) / entry * 100, "exit_reason": "stop_loss",
                })
                in_position = None
                i += 1
                continue

            # цель: откат от пика на retrace_target_pct% от размера пампа
            drawdown_needed = in_position["pump_size_pct"] * (retrace_target_pct / 100)
            drawdown_now_pct = (in_position["peak_price"] - price) / in_position["peak_price"] * 100

            if drawdown_now_pct >= drawdown_needed:
                trades.append({
                    "leg": "short", "entry_idx": in_position["short_entry_idx"],
                    "exit_idx": i, "entry_price": entry, "exit_price": price,
                    "pnl_pct": (entry - price) / entry * 100, "exit_reason": "target",
                })
                in_position = {"phase": "long", "long_entry_idx": i, "long_entry_price": price}

        elif in_position["phase"] == "long":
            entry = in_position["long_entry_price"]
            change_pct = (price - entry) / entry * 100

            if change_pct >= long_target_pct:
                trades.append({
                    "leg": "long", "entry_idx": in_position["long_entry_idx"],
                    "exit_idx": i, "entry_price": entry, "exit_price": price,
                    "pnl_pct": change_pct, "exit_reason": "target",
                })
                in_position = None
            elif change_pct <= -stop_loss_pct:
                trades.append({
                    "leg": "long", "entry_idx": in_position["long_entry_idx"],
                    "exit_idx": i, "entry_price": entry, "exit_price": price,
                    "pnl_pct": change_pct, "exit_reason": "stop_loss",
                })
                in_position = None

        i += 1

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_time"] = trades_df["entry_idx"].apply(lambda idx: times[idx])
        trades_df["exit_time"] = trades_df["exit_idx"].apply(lambda idx: times[idx])
    return trades_df


# ============================================================
# 5. ОСНОВНОЙ ЗАПУСК
# ============================================================

def timeframe_to_hours(tf):
    mapping = {"1m": 1/60, "5m": 5/60, "15m": 15/60, "1h": 1, "4h": 4, "1d": 24}
    return mapping.get(tf, 1)

def main():
    df = fetch_ohlcv_full(EXCHANGE_ID, SYMBOL, TIMEFRAME, SINCE_DATE)
    tf_hours = timeframe_to_hours(TIMEFRAME)

    print("\n=== Поиск эпизодов пампа ===")
    episodes = find_pump_episodes(df, PUMP_THRESHOLD_PCT, PUMP_WINDOW_HOURS, COOLDOWN_HOURS, tf_hours)
    print(f"Найдено эпизодов роста >= {PUMP_THRESHOLD_PCT}% за {PUMP_WINDOW_HOURS}ч: {len(episodes)}")

    if episodes.empty:
        print("Эпизодов не найдено -- попробуйте снизить PUMP_THRESHOLD_PCT или расширить окно.")
        return

    analyzed = analyze_reversion(df, episodes)
    analyzed.to_csv("pump_episodes.csv", index=False)

    print("\n=== Статистика возврата к среднему после пампа ===")
    dd = analyzed["max_drawdown_after_peak_pct"].dropna()
    print(f"Медианная просадка после пика:  {dd.median():.1f}%")
    print(f"Средняя просадка после пика:    {dd.mean():.1f}%")
    print(f"25-й / 75-й перцентиль:         {dd.quantile(0.25):.1f}% / {dd.quantile(0.75):.1f}%")

    rebound = analyzed["second_wave_rebound_pct"].dropna()
    if len(rebound) > 0:
        print(f"\nДоля эпизодов со 'второй волной' роста >= 30%: "
              f"{(rebound >= 30).mean() * 100:.0f}%")
        print(f"Медианный размер второй волны: {rebound.median():.1f}%")

    print("\n=== Бэктест стратегии шорт->лонг ===")
    trades = backtest_short_then_long(
        df, SHORT_ENTRY_PCT, SHORT_TARGET_RETRACE_PCT,
        LONG_TARGET_PCT, STOP_LOSS_PCT, PUMP_WINDOW_HOURS, tf_hours
    )
    trades.to_csv("backtest_trades.csv", index=False)

    if trades.empty:
        print("Сделок не найдено при текущих параметрах.")
        return

    print(f"Всего сделок: {len(trades)}")
    for leg in ["short", "long"]:
        leg_trades = trades[trades["leg"] == leg]
        if leg_trades.empty:
            continue
        win_rate = (leg_trades["pnl_pct"] > 0).mean() * 100
        print(f"\n[{leg.upper()}] сделок: {len(leg_trades)}, "
              f"win rate: {win_rate:.0f}%, "
              f"средний PnL: {leg_trades['pnl_pct'].mean():.1f}%, "
              f"медианный PnL: {leg_trades['pnl_pct'].median():.1f}%")

    total_pnl_pct_sum = trades["pnl_pct"].sum()
    print(f"\nСуммарный (не сложный, без реинвестирования) PnL по всем ногам: {total_pnl_pct_sum:.1f}%")
    print("\nВАЖНО: это упрощённый бэктест без комиссий, funding rate, проскальзывания "
          "и без учёта того, что вход по закрытию бара может не исполниться по этой цене "
          "в реальности (особенно на низколиквидных альтах). Используйте как отправную точку, "
          "не как готовый торговый сигнал.")


if __name__ == "__main__":
    main()
