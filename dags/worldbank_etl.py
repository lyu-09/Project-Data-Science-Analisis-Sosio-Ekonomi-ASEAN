from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pendulum


# ============================================================
# KONFIGURASI
# ============================================================

MYSQL_CONFIG = {
    "host": "host.docker.internal",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "etl_worldbank"
}

COUNTRIES = {
    "IDN": "Indonesia",
    "MYS": "Malaysia",
    "SGP": "Singapore",
    "THA": "Thailand",
    "PHL": "Philippines"
}

START_YEAR = 2015
END_YEAR = 2024


# ============================================================
# SKEMA DATABASE (4 Tabel Relasional)
# ============================================================
#
#   dim_country          <-- Tabel dimensi master negara
#       |
#       +-- fact_economic_data   (FK: country_code)
#       |     Kolom: population, gdp, gdp_per_capita,
#       |            unemployment_rate, life_expectancy
#       |
#       +-- fact_education_data  (FK: country_code)
#       |     Kolom: school_enrollment_primary,
#       |            school_enrollment_secondary,
#       |            literacy_rate_adult,
#       |            govt_expenditure_education
#       |
#       +-- fact_health_data     (FK: country_code)
#             Kolom: health_expenditure_pct_gdp,
#                    health_expenditure_per_capita,
#                    infant_mortality_rate,
#                    hospital_beds_per_1000
#
# ============================================================
# 5 PERTANYAAN ANALITIK
# ============================================================
#
#  Q1 [EKONOMI]:
#     Negara ASEAN mana yang memiliki pertumbuhan GDP per kapita
#     paling konsisten dan tinggi selama 2015-2024?
#     -> Analisis tren time-series per negara
#     -> Model AI: regresi linear/polynomial untuk prediksi 2025-2030
#
#  Q2 [EKONOMI x PENDIDIKAN]:
#     Apakah ada korelasi antara anggaran pendidikan pemerintah
#     (% GDP) dengan tingkat pengangguran di negara-negara ASEAN?
#     -> JOIN fact_economic_data + fact_education_data
#     -> Model AI: analisis korelasi Pearson / regresi linier
#
#  Q3 [KESEHATAN x EKONOMI]:
#     Apakah negara dengan pengeluaran kesehatan per kapita lebih
#     tinggi memiliki angka kematian bayi (infant mortality) lebih
#     rendah? (hipotesis: investasi kesehatan = kualitas hidup)
#     -> JOIN fact_health_data + fact_economic_data
#     -> Model AI: scatter analysis + regresi
#
#  Q4 [MULTI-TABEL - HDI Proxy]:
#     Negara mana yang memiliki kualitas SDM paling baik?
#     Proxy HDI = gabungan: life_expectancy + literacy_rate_adult
#                         + gdp_per_capita (ternormalisasi)
#     -> JOIN ketiga fact table + dim_country
#     -> Model AI: composite scoring dan ranking tahunan
#
#  Q5 [PENDIDIKAN x EKONOMI - LAG ANALYSIS]:
#     Apakah angka partisipasi sekolah menengah pada tahun T
#     berpengaruh terhadap GDP per kapita pada tahun T+3?
#     (investasi pendidikan berdampak tertunda pada ekonomi)
#     -> Self-JOIN fact tables dengan offset tahun
#     -> Model AI: time-lag regression analysis
#
# ============================================================


# ============================================================
# EXTRACT
# ============================================================

def extract_worldbank_data(**context):
    import requests
    import pandas as pd

    print("=== EXTRACT PROCESS STARTED ===")

    indicators_economic = {
        "population":        "SP.POP.TOTL",
        "gdp":               "NY.GDP.MKTP.CD",
        "gdp_per_capita":    "NY.GDP.PCAP.CD",
        "unemployment_rate": "SL.UEM.TOTL.ZS",
        "life_expectancy":   "SP.DYN.LE00.IN"
    }

    indicators_education = {
        "school_enrollment_primary":   "SE.PRM.ENRR",
        "school_enrollment_secondary": "SE.SEC.ENRR",
        "literacy_rate_adult":         "SE.ADT.LITR.ZS",
        "govt_expenditure_education":  "SE.XPD.TOTL.GD.ZS"
    }

    indicators_health = {
        "health_expenditure_pct_gdp":    "SH.XPD.CHEX.GD.ZS",
        "health_expenditure_per_capita": "SH.XPD.CHEX.PC.CD",
        "infant_mortality_rate":         "SP.DYN.IMRT.IN",
        "hospital_beds_per_1000":        "SH.MED.BEDS.ZS"
    }

    def fetch_with_retry(url, max_retries=5, timeout=120):
        import time
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,           # tunggu 2, 4, 8, 16, 32 detik
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://",  adapter)

        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(url, timeout=timeout)
                response.raise_for_status()
                return response
            except Exception as e:
                wait = 2 ** attempt
                print(
                    "  [RETRY " + str(attempt) + "/" + str(max_retries) + "]"
                    " Error: " + str(e)
                    + " | Menunggu " + str(wait) + " detik..."
                )
                if attempt == max_retries:
                    raise
                time.sleep(wait)

    def fetch_indicators(indicator_dict, label):
        all_data = []
        for country_code, country_name in COUNTRIES.items():
            for indicator_name, indicator_code in indicator_dict.items():
                url = (
                    "https://api.worldbank.org/v2/country/"
                    + country_code
                    + "/indicator/"
                    + indicator_code
                    + "?format=json"
                    + "&date=" + str(START_YEAR) + ":" + str(END_YEAR)
                    + "&per_page=100"
                )
                print("[" + label + "] Fetching: " + country_code + " | " + indicator_name)
                response = fetch_with_retry(url, max_retries=5, timeout=120)
                data = response.json()
                if len(data) < 2 or data[1] is None:
                    print("  -> No data returned")
                    continue
                for record in data[1]:
                    all_data.append({
                        "country":      country_name,
                        "country_code": country_code,
                        "year":         record["date"],
                        "indicator":    indicator_name,
                        "value":        record["value"]
                    })
        print("[" + label + "] Total records: " + str(len(all_data)))
        return pd.DataFrame(all_data)

    df_economic  = fetch_indicators(indicators_economic,  "ECONOMIC")
    df_education = fetch_indicators(indicators_education, "EDUCATION")
    df_health    = fetch_indicators(indicators_health,    "HEALTH")

    path_economic  = "/tmp/wb_raw_economic.csv"
    path_education = "/tmp/wb_raw_education.csv"
    path_health    = "/tmp/wb_raw_health.csv"

    df_economic.to_csv(path_economic,   index=False)
    df_education.to_csv(path_education, index=False)
    df_health.to_csv(path_health,       index=False)

    context["ti"].xcom_push(key="path_economic",  value=path_economic)
    context["ti"].xcom_push(key="path_education", value=path_education)
    context["ti"].xcom_push(key="path_health",    value=path_health)

    print("=== EXTRACT PROCESS FINISHED ===")


# ============================================================
# TRANSFORM
# ============================================================

def transform_data(**context):
    import pandas as pd

    print("=== TRANSFORM PROCESS STARTED ===")

    ti = context["ti"]

    path_economic  = ti.xcom_pull(task_ids="extract_data", key="path_economic")
    path_education = ti.xcom_pull(task_ids="extract_data", key="path_education")
    path_health    = ti.xcom_pull(task_ids="extract_data", key="path_health")

    def pivot_and_clean(csv_path, label):
        df = pd.read_csv(csv_path)
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["year", "value"])
        df = df.pivot_table(
            index=["country", "country_code", "year"],
            columns="indicator",
            values="value"
        ).reset_index()
        df.columns.name = None
        df = df.sort_values(by=["country", "year"]).reset_index(drop=True)
        print("[" + label + "] Rows after transform: " + str(len(df)))
        return df

    df_eco = pivot_and_clean(path_economic,  "ECONOMIC")
    df_edu = pivot_and_clean(path_education, "EDUCATION")
    df_hlt = pivot_and_clean(path_health,    "HEALTH")

    # --------------------------------------------------------
    # Kolom turunan untuk Economic
    # --------------------------------------------------------

    if "population" in df_eco.columns:
        df_eco["population_million"] = (df_eco["population"] / 1_000_000).round(2)

    if "gdp" in df_eco.columns:
        df_eco["gdp_billion"] = (df_eco["gdp"] / 1_000_000_000).round(2)

    for col in ["gdp_per_capita", "unemployment_rate", "life_expectancy"]:
        if col in df_eco.columns:
            df_eco[col] = df_eco[col].round(2)

    for col in ["school_enrollment_primary", "school_enrollment_secondary",
                "literacy_rate_adult", "govt_expenditure_education"]:
        if col in df_edu.columns:
            df_edu[col] = df_edu[col].round(2)

    for col in ["health_expenditure_pct_gdp", "health_expenditure_per_capita",
                "infant_mortality_rate", "hospital_beds_per_1000"]:
        if col in df_hlt.columns:
            df_hlt[col] = df_hlt[col].round(2)

    # --------------------------------------------------------
    # Buat dim_country (union negara dari semua fact table)
    # --------------------------------------------------------

    all_countries = (
        pd.concat([
            df_eco[["country", "country_code"]],
            df_edu[["country", "country_code"]],
            df_hlt[["country", "country_code"]]
        ])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    region_map = {
        "IDN": "South-East Asia",
        "MYS": "South-East Asia",
        "SGP": "South-East Asia",
        "THA": "South-East Asia",
        "PHL": "South-East Asia"
    }
    income_map = {
        "IDN": "Lower-middle income",
        "MYS": "Upper-middle income",
        "SGP": "High income",
        "THA": "Upper-middle income",
        "PHL": "Lower-middle income"
    }

    all_countries["region"]       = all_countries["country_code"].map(region_map)
    all_countries["income_group"] = all_countries["country_code"].map(income_map)

    print("[DIM_COUNTRY] Rows: " + str(len(all_countries)))

    # --------------------------------------------------------
    # Simpan semua hasil transform ke CSV
    # --------------------------------------------------------

    path_dim_country = "/tmp/wb_dim_country.csv"
    path_fact_eco    = "/tmp/wb_fact_economic.csv"
    path_fact_edu    = "/tmp/wb_fact_education.csv"
    path_fact_hlt    = "/tmp/wb_fact_health.csv"

    all_countries.to_csv(path_dim_country, index=False)
    df_eco.to_csv(path_fact_eco,           index=False)
    df_edu.to_csv(path_fact_edu,           index=False)
    df_hlt.to_csv(path_fact_hlt,           index=False)

    ti.xcom_push(key="path_dim_country", value=path_dim_country)
    ti.xcom_push(key="path_fact_eco",    value=path_fact_eco)
    ti.xcom_push(key="path_fact_edu",    value=path_fact_edu)
    ti.xcom_push(key="path_fact_hlt",    value=path_fact_hlt)

    print("=== TRANSFORM PROCESS FINISHED ===")


# ============================================================
# LOAD
# ============================================================

def load_to_mysql(**context):
    import pandas as pd
    import mysql.connector

    print("=== LOAD PROCESS STARTED ===")

    ti = context["ti"]

    path_dim_country = ti.xcom_pull(task_ids="transform_data", key="path_dim_country")
    path_fact_eco    = ti.xcom_pull(task_ids="transform_data", key="path_fact_eco")
    path_fact_edu    = ti.xcom_pull(task_ids="transform_data", key="path_fact_edu")
    path_fact_hlt    = ti.xcom_pull(task_ids="transform_data", key="path_fact_hlt")

    df_country = pd.read_csv(path_dim_country)
    df_eco     = pd.read_csv(path_fact_eco)
    df_edu     = pd.read_csv(path_fact_edu)
    df_hlt     = pd.read_csv(path_fact_hlt)

    # --------------------------------------------------------
    # Connect ke MySQL
    # --------------------------------------------------------

    connection = mysql.connector.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"]
    )
    cursor = connection.cursor()
    print("Connected to MySQL")

    # --------------------------------------------------------
    # DDL: Buat semua tabel (nonaktifkan FK sementara)
    # --------------------------------------------------------

    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    connection.commit()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_country (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            country_code VARCHAR(10)  NOT NULL UNIQUE,
            country      VARCHAR(100) NOT NULL,
            region       VARCHAR(100),
            income_group VARCHAR(100),
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_economic_data (
            id                 INT AUTO_INCREMENT PRIMARY KEY,
            country_code       VARCHAR(10) NOT NULL,
            year               INT         NOT NULL,
            population         BIGINT,
            population_million DECIMAL(15,2),
            gdp                DECIMAL(30,2),
            gdp_billion        DECIMAL(20,2),
            gdp_per_capita     DECIMAL(20,2),
            unemployment_rate  DECIMAL(10,2),
            life_expectancy    DECIMAL(10,2),
            created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_eco (country_code, year),
            CONSTRAINT fk_eco_country FOREIGN KEY (country_code)
                REFERENCES dim_country (country_code) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_education_data (
            id                          INT AUTO_INCREMENT PRIMARY KEY,
            country_code                VARCHAR(10) NOT NULL,
            year                        INT         NOT NULL,
            school_enrollment_primary   DECIMAL(10,2),
            school_enrollment_secondary DECIMAL(10,2),
            literacy_rate_adult         DECIMAL(10,2),
            govt_expenditure_education  DECIMAL(10,2),
            created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_edu (country_code, year),
            CONSTRAINT fk_edu_country FOREIGN KEY (country_code)
                REFERENCES dim_country (country_code) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_health_data (
            id                            INT AUTO_INCREMENT PRIMARY KEY,
            country_code                  VARCHAR(10) NOT NULL,
            year                          INT         NOT NULL,
            health_expenditure_pct_gdp    DECIMAL(10,2),
            health_expenditure_per_capita DECIMAL(15,2),
            infant_mortality_rate         DECIMAL(10,2),
            hospital_beds_per_1000        DECIMAL(10,2),
            created_at                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_hlt (country_code, year),
            CONSTRAINT fk_hlt_country FOREIGN KEY (country_code)
                REFERENCES dim_country (country_code) ON DELETE CASCADE
        )
    """)

    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    connection.commit()
    print("All 4 tables ready")

    # --------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------

    def safe_float(row, col):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
        return None

    def safe_int(row, col):
        if col in row.index and pd.notna(row[col]):
            return int(row[col])
        return None

    # --------------------------------------------------------
    # INSERT dim_country
    # --------------------------------------------------------

    for _, row in df_country.iterrows():
        cursor.execute("""
            INSERT INTO dim_country (country_code, country, region, income_group)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                country=VALUES(country),
                region=VALUES(region),
                income_group=VALUES(income_group)
        """, (
            row["country_code"], row["country"],
            row.get("region", None), row.get("income_group", None)
        ))
    connection.commit()
    print("Loaded " + str(len(df_country)) + " rows into dim_country")

    # --------------------------------------------------------
    # INSERT fact_economic_data
    # --------------------------------------------------------

    for _, row in df_eco.iterrows():
        cursor.execute("""
            INSERT INTO fact_economic_data (
                country_code, year,
                population, population_million,
                gdp, gdp_billion, gdp_per_capita,
                unemployment_rate, life_expectancy
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                population=VALUES(population),
                population_million=VALUES(population_million),
                gdp=VALUES(gdp),
                gdp_billion=VALUES(gdp_billion),
                gdp_per_capita=VALUES(gdp_per_capita),
                unemployment_rate=VALUES(unemployment_rate),
                life_expectancy=VALUES(life_expectancy)
        """, (
            row["country_code"], int(row["year"]),
            safe_int(row,   "population"),
            safe_float(row, "population_million"),
            safe_float(row, "gdp"),
            safe_float(row, "gdp_billion"),
            safe_float(row, "gdp_per_capita"),
            safe_float(row, "unemployment_rate"),
            safe_float(row, "life_expectancy")
        ))
    connection.commit()
    print("Loaded " + str(len(df_eco)) + " rows into fact_economic_data")

    # --------------------------------------------------------
    # INSERT fact_education_data
    # --------------------------------------------------------

    for _, row in df_edu.iterrows():
        cursor.execute("""
            INSERT INTO fact_education_data (
                country_code, year,
                school_enrollment_primary,
                school_enrollment_secondary,
                literacy_rate_adult,
                govt_expenditure_education
            ) VALUES (%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                school_enrollment_primary=VALUES(school_enrollment_primary),
                school_enrollment_secondary=VALUES(school_enrollment_secondary),
                literacy_rate_adult=VALUES(literacy_rate_adult),
                govt_expenditure_education=VALUES(govt_expenditure_education)
        """, (
            row["country_code"], int(row["year"]),
            safe_float(row, "school_enrollment_primary"),
            safe_float(row, "school_enrollment_secondary"),
            safe_float(row, "literacy_rate_adult"),
            safe_float(row, "govt_expenditure_education")
        ))
    connection.commit()
    print("Loaded " + str(len(df_edu)) + " rows into fact_education_data")

    # --------------------------------------------------------
    # INSERT fact_health_data
    # --------------------------------------------------------

    for _, row in df_hlt.iterrows():
        cursor.execute("""
            INSERT INTO fact_health_data (
                country_code, year,
                health_expenditure_pct_gdp,
                health_expenditure_per_capita,
                infant_mortality_rate,
                hospital_beds_per_1000
            ) VALUES (%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                health_expenditure_pct_gdp=VALUES(health_expenditure_pct_gdp),
                health_expenditure_per_capita=VALUES(health_expenditure_per_capita),
                infant_mortality_rate=VALUES(infant_mortality_rate),
                hospital_beds_per_1000=VALUES(hospital_beds_per_1000)
        """, (
            row["country_code"], int(row["year"]),
            safe_float(row, "health_expenditure_pct_gdp"),
            safe_float(row, "health_expenditure_per_capita"),
            safe_float(row, "infant_mortality_rate"),
            safe_float(row, "hospital_beds_per_1000")
        ))
    connection.commit()
    print("Loaded " + str(len(df_hlt)) + " rows into fact_health_data")

    # --------------------------------------------------------
    # Close connection
    # --------------------------------------------------------

    cursor.close()
    connection.close()
    print("MySQL connection closed")
    print("=== LOAD PROCESS FINISHED ===")


# ============================================================
# AIRFLOW DAG
# ============================================================

default_args = {
    "owner":            "lyu",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5)
}

with DAG(
    dag_id="worldbank_etl_multi_table",
    default_args=default_args,
    description=(
        "ETL World Bank API -> MySQL | "
        "4 tabel relasional: dim_country + Ekonomi + Pendidikan + Kesehatan | "
        "5 negara ASEAN 2015-2024"
    ),
    start_date=datetime(2024, 1, 1, tzinfo=pendulum.timezone("UTC")),
    schedule=None,
    catchup=False,
    tags=["ETL", "WorldBank", "MySQL", "ASEAN", "Multi-Table"]
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_worldbank_data
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data
    )

    load_task = PythonOperator(
        task_id="load_to_mysql",
        python_callable=load_to_mysql
    )

    # ========================================================
    # PIPELINE: Extract -> Transform -> Load
    # ========================================================

    extract_task >> transform_task >> load_task
