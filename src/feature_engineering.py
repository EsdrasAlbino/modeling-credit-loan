import json
from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path("data")
DATASET_DIR = Path("dataset")


def safe_midpoint(val):
    if pd.isna(val):
        return np.nan
    try:
        # handle strings like '150-250' or '150 - 250'
        if isinstance(val, str) and ("-" in val):
            parts = [p.strip() for p in val.split("-")]
            nums = [float(p) for p in parts if p.replace('.', '', 1).isdigit()]
            if len(nums) == 2:
                return sum(nums) / 2.0
        # otherwise coerce to numeric
        return float(val)
    except Exception:
        return np.nan


def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def build_application_features(df):
    df_out = df.copy()

    # Example boolean flags
    if 'serasa_contagem_negativacoes' in df_out.columns:
        df_out['has_serasa_negativacao'] = df_out['serasa_contagem_negativacoes'].fillna(0) > 0
    else:
        df_out['has_serasa_negativacao'] = False

    # ifood
    if 'ifood_contagem_avaliacoes' in df_out.columns:
        df_out['has_ifood'] = df_out['ifood_contagem_avaliacoes'].notna()
    elif 'ifood' in df_out.columns:
        df_out['has_ifood'] = df_out['ifood'].notna()
    else:
        df_out['has_ifood'] = False

    # google maps
    gm_cols = [c for c in df_out.columns if 'google' in c.lower()]
    if gm_cols:
        df_out['has_google_maps'] = df_out[gm_cols].notna().any(axis=1)
    else:
        df_out['has_google_maps'] = False

    # idade_cnpj midpoint / numeric
    if 'idade_cnpj' in df_out.columns:
        df_out['idade_cnpj_midpoint'] = df_out['idade_cnpj'].apply(safe_midpoint)
    else:
        df_out['idade_cnpj_midpoint'] = np.nan

    # basic imputation for numeric columns
    num_cols = df_out.select_dtypes(include=['number']).columns.tolist()
    for c in num_cols:
        df_out[c] = df_out[c].fillna(df_out[c].median())

    # one-hot a few low-cardinality categoricals
    cat_candidates = ['uf', 'segmento_cliente', 'natureza_juridica', 'fonte_cliente']
    to_onehot = [c for c in cat_candidates if c in df_out.columns]
    if to_onehot:
        df_out = pd.get_dummies(df_out, columns=to_onehot, dummy_na=False, drop_first=True)

    return df_out


def aggregate_behavioral(df_orders):
    df = df_orders.copy()
    # attempt to find id and value/date/delay columns
    id_col = None
    for candidate in ['id_cliente', 'cliente_id', 'id_cliente_pedido', 'cliente']:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError('Could not find client id column in orders dataset')

    # value column heuristics
    value_col = None
    for candidate in ['valor', 'valor_pedido', 'valor_total', 'valor_final', 'total']:
        if candidate in df.columns:
            value_col = candidate
            break

    # delay column heuristics
    delay_col = None
    for candidate in ['atraso', 'dias_atraso', 'delay_days', 'dias_de_atraso']:
        if candidate in df.columns:
            delay_col = candidate
            break

    # date column heuristics
    date_col = None
    for candidate in ['data_pedido', 'data', 'pedido_data', 'created_at']:
        if candidate in df.columns:
            date_col = candidate
            break

    agg = df.groupby(id_col).agg(
        orders_count=(id_col, 'size')
    )

    if value_col:
        # coerce to numeric (handle strings, currency symbols, commas)
        df[value_col] = (
            df[value_col]
            .astype(str)
            .str.replace(r"[^0-9,.-]", "", regex=True)
            .str.replace(',', '.', regex=False)
        )
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        agg_val = df.groupby(id_col)[value_col].agg(['min', 'max', 'mean', 'median', 'std']).add_prefix('valor_')
        agg = agg.join(agg_val)

    if delay_col:
        # coerce delay to numeric
        df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')
        agg_delay = df.groupby(id_col)[delay_col].agg(['mean', 'max'])
        agg_delay.columns = ['delay_mean', 'delay_max']
        agg = agg.join(agg_delay)
        # percent delayed
        agg_pct = df.assign(is_delayed=lambda x: (x[delay_col].fillna(0) > 0)).groupby(id_col)['is_delayed'].mean().rename('pct_orders_delayed')
        agg = agg.join(agg_pct)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        last = df.groupby(id_col)[date_col].max().rename('last_order_date')
        first = df.groupby(id_col)[date_col].min().rename('first_order_date')
        agg = agg.join(last).join(first)
        agg['recency_days'] = (pd.Timestamp.now() - agg['last_order_date']).dt.days
        # orders per month approximate: orders_count / months_active
        months_active = ((pd.to_datetime(agg['last_order_date']) - pd.to_datetime(agg['first_order_date'])).dt.days / 30).replace(0, 1)
        agg['orders_per_month'] = agg['orders_count'] / months_active

    agg = agg.reset_index()
    return agg


def main():
    DATA_DIR.mkdir(exist_ok=True)

    app_path = DATASET_DIR / 'credito_aplicacao_clientes_final.csv'
    orders_path = DATASET_DIR / 'credito_comportamental_pedidos_final.csv'

    if not app_path.exists():
        print(f'Missing application dataset: {app_path}')
        return
    if not orders_path.exists():
        print(f'Missing orders dataset: {orders_path}')
        return

    print('Loading application data...')
    df_app = load_csv(app_path)
    print(f'Application rows: {len(df_app)}, columns: {len(df_app.columns)}')

    print('Building application features...')
    df_app_feats = build_application_features(df_app)
    app_out = DATA_DIR / '03_application_features.csv'
    df_app_feats.to_csv(app_out, index=False)
    print(f'Wrote application features to {app_out}')

    print('Loading orders data...')
    df_orders = load_csv(orders_path)
    print(f'Orders rows: {len(df_orders)}, columns: {len(df_orders.columns)}')

    print('Aggregating behavioral features...')
    try:
        df_beh = aggregate_behavioral(df_orders)
        beh_out = DATA_DIR / '03_behavioral_aggregates.csv'
        df_beh.to_csv(beh_out, index=False)
        print(f'Wrote behavioral aggregates to {beh_out}')
    except Exception as e:
        print('Behavioral aggregation failed:', e)
        return

    # attempt join
    join_keys = ['id_cliente', 'cliente_id', 'id']
    join_key = None
    for k in join_keys:
        if k in df_app_feats.columns and k in df_beh.columns:
            join_key = k
            break
    if join_key is None:
        # try matching on columns with similar names
        possible = [c for c in df_app_feats.columns if any(s in c for s in ['cliente','id'])]
        if possible and 'id_cliente' in df_beh.columns:
            join_key = 'id_cliente'

    if join_key:
        combined = df_app_feats.merge(df_beh, how='left', left_on=join_key, right_on=join_key)
        combined_out = DATA_DIR / '03_behavioral_combined.csv'
        combined.to_csv(combined_out, index=False)
        print(f'Wrote combined dataset to {combined_out}')
    else:
        print('Could not determine join key between application and behavioral datasets; saved separate outputs.')


if __name__ == '__main__':
    main()
