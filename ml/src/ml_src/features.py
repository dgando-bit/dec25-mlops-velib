import pandas as pd

_SEVERITY = {
	**{c: 0 for c in [0, 1]},
	**{c: 1 for c in [2, 3, 45, 48]},
	**{c: 2 for c in [51, 53, 55, 61]},
}


def _add_weather_severity(df: pd.DataFrame) -> pd.DataFrame:
	df['weather_severity'] = (
		df['weather_code'].map(_SEVERITY).fillna(3).astype(int)
	)
	return df


# def preprocess_weather(df):
# 	def get_severity(code):
# 		if code in [0, 1]: return 0  # Ciel clair / Peu nuageux
# 		if code in [2, 3, 45, 48]: return 1  # Nuageux / Brouillard
# 		if code in [51, 53, 55, 61]: return 2  # Bruine / Pluie légère
# 		return 3  # Pluie forte, Neige, Orage (Le chaos)
#
# 	df['weather_severity'] = df['weather_code'].apply(get_severity).astype('category')
# 	return df


def clean_velib_data(df: pd.DataFrame) -> pd.DataFrame:
	cols = ['is_renting', 'is_holiday', 'is_vacation']

	df = df.rename(columns={
		"datetime": "last_reported",
		"numdocksavailable": "num_docks_available"
	})

	# df = preprocess_weather(df)
	df = _add_weather_severity(df)

	df['last_reported'] = pd.to_datetime(df['last_reported'])
	df['hour'] = df['last_reported'].dt.hour

	# for col in cols:
	# 	print(col, df[col].unique())

	# df[cols] = df[cols].apply(lambda col: col.map({'t': 0, 'f': 1})).astype(int)
	df[cols] = df[cols].astype(int)
	return df


def resample_data(df):
	# Conversion de la date
	df['last_reported'] = pd.to_datetime(df['last_reported'])
	df['last_reported_h'] = df['last_reported'].dt.floor('h')

	# Agrégation par heure
	df_resampled = df.groupby(['station_id', 'last_reported_h']).agg({
		'capacity_status': 'mean',
		'name': 'first',
		'apparent_temperature': 'mean',
		'weather_code': 'first',
		'weather_severity': 'first',
		'is_holiday': 'first',
		'is_vacation': 'first',
		'lat': 'first',
		'lon': 'first'
	}).reset_index()

	# Resampling
	df_resampled = df_resampled.set_index('last_reported_h')

	# On groupe et on rééchantillonne
	df_resampled = (
		df_resampled.groupby('station_id', group_keys=False)  # group_keys=False aide à la structure finale
		.resample('h')
		.first()
	)

	# Si 'station_id' est dans l'index, il reviendra en colonne automatiquement.
	df_resampled = df_resampled.reset_index()

	# Si Pandas a créé un doublon ou nommé la colonne différemment (ex: level_0),
	# on nettoie dynamiquement :
	if 'level_0' in df_resampled.columns:
		df_resampled = df_resampled.rename(columns={'level_0': 'station_id'})

	# On vérifie qu'on n'a pas deux colonnes station_id
	df_resampled = df_resampled.loc[:, ~df_resampled.columns.duplicated()]

	# 4. Remplissage des vides
	# Interpolation linéaire pour boucher les trous du taux d'occupation
	df_resampled['capacity_status'] = df_resampled.groupby('station_id')['capacity_status'].transform(
		lambda x: x.interpolate(method='linear'))

	# Remplissage des données fixes (météo, coordonnées...)
	cols_to_fill = ['apparent_temperature', 'weather_code', 'weather_severity', 'is_holiday', 'is_vacation', 'lat', 'lon', 'name']
	df_resampled[cols_to_fill] = df_resampled.groupby('station_id')[cols_to_fill].ffill()

	# Variables de temps, Lags
	df_resampled['hour'] = df_resampled['last_reported_h'].dt.hour
	df_resampled['day_of_week'] = df_resampled['last_reported_h'].dt.dayofweek

	df_resampled['capacity_status_lag_1h'] = df_resampled.groupby('station_id')['capacity_status'].shift(1)
	df_resampled['capacity_status_lag_2h'] = df_resampled.groupby('station_id')['capacity_status'].shift(2)
	df_resampled['capacity_status_lag_24h'] = df_resampled.groupby('station_id')['capacity_status'].shift(24)

	# Nettoyage final
	df_final = df_resampled.dropna(
		subset=['capacity_status_lag_1h', 'capacity_status_lag_2h', 'capacity_status_lag_24h'])

	return df_final


def add_trend_features(df):
	df = df.copy()
	# Tendance immédiate : Variation entre T-1h et T-2h
	df['diff_1h_2h'] = df['capacity_status_lag_1h'] - df['capacity_status_lag_2h']

	# Tendance journalière : Variation entre T-1h et T-24h
	# (Est-ce qu'on est plus plein ou plus vide qu'hier à la même heure ?)
	df['diff_1h_24h'] = df['capacity_status_lag_1h'] - df['capacity_status_lag_24h']

	return df


def build_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
	df = clean_velib_data(df)
	df = resample_data(df)
	df = add_trend_features(df)
	return df
