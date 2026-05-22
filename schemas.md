# Velib MLOps - Schemas d'architecture

## Architecture des conteneurs — 01_containers

```mermaid
flowchart TD
    internet([Internet])

    subgraph proxy[Reverse Proxy]
        nginx[nginx]
    end

    subgraph ml[ML Stack]
        api[api FastAPI]
        mlflow_srv[mlflow-server]
        mlflow_db[(mlflow-db Postgres 15)]
        ml_train[ml_training one-shot]
        jupyter[jupyter-service]
    end

    subgraph obs[Monitoring]
        prometheus[prometheus]
        grafana[grafana]
        node_exp[node-exporter]
    end

    subgraph orch[Airflow CeleryExecutor]
        aw[webserver]
        sched[scheduler]
        worker[worker]
        redis[(redis)]
        adb[(airflow-db Postgres 15)]
        flower[flower]
        ainit[init one-shot]
    end

    streamlit[streamlit]

    internet --> nginx
    nginx -->|8080 api| api
    nginx -->|5000 mlflow| mlflow_srv
    nginx -->|8888 jupyter| jupyter
    nginx -->|9090 prometheus| prometheus
    nginx -->|3000 grafana| grafana
    nginx -->|8090 airflow| aw
    nginx -->|5555 flower| flower
    nginx -->|8501 streamlit| streamlit

    api --> mlflow_srv
    mlflow_srv --> mlflow_db
    ml_train -->|log runs| mlflow_srv
    api -->|metrics| prometheus
    node_exp --> prometheus
    prometheus --> grafana
    sched -->|dispatch| redis
    worker -->|consume| redis
    worker -->|log runs| mlflow_srv
    sched --> adb
    ainit --> adb
    flower -->|monitor| redis
```

---

## Pipeline ML de bout en bout — 02_pipeline

```mermaid
flowchart TD
    hf[(HuggingFace)]

    load[load_from_hf]
    make[make_dataset]
    viz[dataviz rapport HTML]
    build[build_features]
    train[train_model XGBoost]
    detect[detect_drift Evidently]
    drift_out[drift_metrics.json]
    registry[(MLflow Registry alias staging)]
    api[api FastAPI]
    client([Client])

    hf -->|CSV snapshots| load
    load -->|raw parquet| make
    make -->|cleaned parquet| build
    make --> viz
    build -->|train-test parquet| train
    build -->|train-test parquet| detect
    detect --> drift_out
    train -->|register model| registry
    registry -->|load at startup| api
    api -->|POST /predict| client
```

---

## Flux de données & artefacts — 03_data_flow

```mermaid
flowchart LR
    hf[(HuggingFace)]

    subgraph data[data/ bind mount DVC]
        raw[data/raw parquet brut]
        interim[data/interim parquet nettoye]
        proc[data/processed train test stations]
        out_m[outputs/ metrics.json]
        out_d[outputs/drift drift_metrics.json]
        out_p[outputs/plots dataviz.html]
    end

    subgraph mlf[MLflow Backend]
        mlf_db[(mlflow-db Postgres 15)]
        mlf_art[mlflow/artifacts bind mount]
        mlf_reg[(MLflow Registry)]
    end

    api[api FastAPI]

    hf --> raw
    raw --> interim
    interim --> proc
    interim --> out_p
    proc --> out_m
    proc --> out_d
    proc -->|run metadata| mlf_db
    proc -->|model artifact| mlf_art
    mlf_db --> mlf_reg
    mlf_art --> mlf_reg
    mlf_reg -->|alias staging| api
```

---

## Requête d'inférence — 04_inference

```mermaid
flowchart TD
    client([Client])
    nginx[nginx rate limit 10r/s]
    api[api FastAPI]
    model_check{model en cache}
    mlflow_load[mlflow-server charge modele]
    lru[lru_cache stocke modele]
    predict[predict_with_confidence 24 features]
    reconstruct[taux = trend + residuel clip 0-100]
    alert_check{alert_level}
    green[green taux 30 a 70]
    yellow[yellow taux 10-30 ou 70-90]
    red[red taux sous 10 ou sur 90]
    response[200 JSON taux residuel alert]

    client -->|POST /predict| nginx
    nginx -->|proxy| api
    api --> model_check
    model_check -->|non - au demarrage| mlflow_load
    mlflow_load --> lru
    lru --> predict
    model_check -->|oui - cache LRU| predict
    predict --> reconstruct
    reconstruct --> alert_check
    alert_check --> green
    alert_check --> yellow
    alert_check --> red
    green --> response
    yellow --> response
    red --> response
    response --> client
```

---

## Observabilité — 05_observability

```mermaid
flowchart TD
    subgraph sources[Sources de metriques]
        api_m[api /metrics Prometheus]
        node_e[node-exporter CPU RAM disque]
        prom_s[prometheus self-scrape 9090]
    end

    subgraph prom_layer[Prometheus scrape 15s]
        j_api[job velib-api api:8000]
        j_node[job node-exporter 9100]
        j_prom[job prometheus 9090]
        tsdb[(TSDB retention 15j)]
    end

    subgraph graf_layer[Grafana]
        ds[datasource Prometheus auto-provisionnee]
        dash[Infra-dashboard CPU memoire reseau]
        a_cpu[alerte CPU over 90pct pendant 5min]
        a_ram[alerte RAM over 80pct pendant 2min]
    end

    contact[contact point receiver empty]

    api_m --> j_api
    node_e --> j_node
    prom_s --> j_prom
    j_api --> tsdb
    j_node --> tsdb
    j_prom --> tsdb
    tsdb --> ds
    ds --> dash
    ds --> a_cpu
    ds --> a_ram
    a_cpu --> contact
    a_ram --> contact
```

---

## Vue d'ensemble — 00_overview

```mermaid
flowchart TD
    client([Client / Jury])

    subgraph src[Source de donnees]
        hf[(HuggingFace)]
    end

    subgraph pipeline[Pipeline ML]
        airflow[Airflow DAG toutes les 6h]
        dvc[DVC 6 stages]
        mlflow[(MLflow Registry)]
        airflow --> dvc
        dvc -->|register alias staging| mlflow
    end

    subgraph serving[Serving]
        nginx[nginx reverse proxy]
        api[api FastAPI]
        streamlit[Streamlit demo]
    end

    subgraph observ[Observabilite]
        prometheus[prometheus]
        grafana[grafana]
        prometheus --> grafana
    end

    hf -->|ingestion| dvc
    mlflow -->|load model| api
    client --> nginx
    nginx --> api
    nginx --> streamlit
    api -->|metrics| prometheus
```
