import math

STATIONS: list[dict] = [
    {"station_id": 6298,  "name": "Gare du Nord",             "lat": 48.8810, "lon": 2.3524, "capacity": 24, "station_trend_avg": 55.0, "morning_evening_ratio": 1.30},
    {"station_id": 6245,  "name": "Ventadour - Opéra",        "lat": 48.8668, "lon": 2.3344, "capacity": 27, "station_trend_avg": 48.0, "morning_evening_ratio": 1.10},
    {"station_id": 6293,  "name": "Mairie du 2ème",           "lat": 48.8672, "lon": 2.3405, "capacity": 28, "station_trend_avg": 53.0, "morning_evening_ratio": 1.20},
    {"station_id": 6352,  "name": "Saint-Bon - Rivoli",       "lat": 48.8580, "lon": 2.3502, "capacity": 31, "station_trend_avg": 50.0, "morning_evening_ratio": 1.00},
    {"station_id": 6296,  "name": "Institut de France",       "lat": 48.8576, "lon": 2.3358, "capacity": 34, "station_trend_avg": 46.0, "morning_evening_ratio": 1.05},
    {"station_id": 6297,  "name": "Pigalle",                  "lat": 48.8811, "lon": 2.3367, "capacity": 18, "station_trend_avg": 42.0, "morning_evening_ratio": 0.80},
    {"station_id": 6295,  "name": "Pontoise - La Tournelle",  "lat": 48.8505, "lon": 2.3525, "capacity": 23, "station_trend_avg": 44.0, "morning_evening_ratio": 1.00},
    {"station_id": 9020,  "name": "République",               "lat": 48.8672, "lon": 2.3630, "capacity": 35, "station_trend_avg": 47.0, "morning_evening_ratio": 1.05},
    {"station_id": 8002,  "name": "Châtelet - Les Halles",    "lat": 48.8603, "lon": 2.3465, "capacity": 38, "station_trend_avg": 56.0, "morning_evening_ratio": 1.00},
    {"station_id": 14002, "name": "Bastille - Arsenal",       "lat": 48.8534, "lon": 2.3685, "capacity": 40, "station_trend_avg": 49.0, "morning_evening_ratio": 1.00},
    {"station_id": 21101, "name": "Montparnasse - Gare",      "lat": 48.8422, "lon": 2.3201, "capacity": 45, "station_trend_avg": 58.0, "morning_evening_ratio": 1.15},
    {"station_id": 16107, "name": "Tour Eiffel - Champ de Mars","lat": 48.8566,"lon": 2.2960,"capacity": 35, "station_trend_avg": 52.0, "morning_evening_ratio": 0.90},
    {"station_id": 10001, "name": "Oberkampf - Parmentier",   "lat": 48.8651, "lon": 2.3789, "capacity": 30, "station_trend_avg": 45.0, "morning_evening_ratio": 0.95},
    {"station_id": 13001, "name": "Place d'Italie",           "lat": 48.8317, "lon": 2.3535, "capacity": 32, "station_trend_avg": 50.0, "morning_evening_ratio": 1.10},
    {"station_id": 17001, "name": "Trocadéro",                "lat": 48.8637, "lon": 2.2889, "capacity": 29, "station_trend_avg": 43.0, "morning_evening_ratio": 0.85},
]

STATIONS_BY_NAME = {s["name"]: s for s in STATIONS}
STATION_NAMES = [s["name"] for s in STATIONS]


def _capacity_group(cap: int) -> int:
    if cap < 15:
        return 0
    elif cap < 25:
        return 1
    elif cap < 40:
        return 2
    return 3


def build_features(
    station: dict,
    hour: int,
    dow: int,
    month: int,
    temperature: float,
    temp_anomalie: float,
    weather_severity: int,
    is_frozen: int,
    is_stormy: int,
    lag_60min: float,
    lag_240min: float,
    is_holiday: int,
    is_vacation: int,
) -> dict:
    trend = station["station_trend_avg"]
    h_sin = math.sin(2 * math.pi * hour / 24)
    h_cos = math.cos(2 * math.pi * hour / 24)
    d_sin = math.sin(2 * math.pi * dow / 7)
    d_cos = math.cos(2 * math.pi * dow / 7)
    is_peak = int(hour in {7, 8, 9, 17, 18, 19})
    is_fri_eve = int(dow == 4 and hour in {17, 18, 19, 20})
    is_mon_morn = int(dow == 0 and hour in {7, 8, 9})
    lag_res_240 = round(lag_240min - trend, 2)

    return {
        "capacity": station["capacity"],
        "capacity_group": _capacity_group(station["capacity"]),
        "morning_evening_ratio": station["morning_evening_ratio"],
        "hour_sin": round(h_sin, 6),
        "hour_cos": round(h_cos, 6),
        "dow_sin": round(d_sin, 6),
        "dow_cos": round(d_cos, 6),
        "month": month,
        "is_peak_hour": is_peak,
        "is_friday_evening": is_fri_eve,
        "is_monday_morning": is_mon_morn,
        "is_holiday": is_holiday,
        "is_vacation": is_vacation,
        "apparent_temperature": temperature,
        "temp_anomalie": temp_anomalie,
        "weather_severity": weather_severity,
        "is_frozen": is_frozen,
        "is_stormy": is_stormy,
        "lag_60min": lag_60min,
        "lag_240min": lag_240min,
        "lag_res_240min": lag_res_240,
        "lat": station["lat"],
        "lon": station["lon"],
        "hour": hour,
        "station_trend_avg": trend,
    }
