import pandas as pd
from impala.dbapi import connect

def get_impala_connection():
    try:
        conn = connect(
            host='dl-dev-cl-dn01.datalake-dev.local',
            port=21050,
            auth_mechanism='GSSAPI',
            kerberos_service_name='impala'
        )
        return conn
    except Exception as e:
        print(f"Impala Connection Error: {e}")
        return None

def fetch_data(query: str) -> pd.DataFrame:
    print("Running on Server. Connecting to Impala via GSSAPI...")
    conn = get_impala_connection()
    if not conn:
        return pd.DataFrame() 
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        cursor.close()
        conn.close()
        return df
    except Exception as e:
        print(f"Impala Query Execution Error: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()