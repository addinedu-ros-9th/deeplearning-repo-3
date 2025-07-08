from dbutils.pooled_db import PooledDB
import pymysql

class DBConnector:
    def __init__(self,
                 host = 'localhost',
                 user = 'kth',
                 password = '*******',
                 database = 'crash_db',
                 max_connections = 5): 
        self.db_pool = PooledDB(
                creator = pymysql,
                maxconnections = max_connections,   #한 번에 최대 몇개의 커넥션을 풀에 보관 할 수 있는 지를 의미/ 이 숫자 만큼만 DB와 연결을 만들고 재사용함.
                blocking = True,                   #풀에 있는 커넥션들이 모두 사용 중일 때, 다음요청이 커넥션을 얻을 때까지 기다리게 만드는 옵션
                #timeout = 0.1,                     #최대 대기 시간 : 0.1초
                host = host,
                user = user,
                password = password,
                database = database,
                charset = 'utf8mb4',
                autocommit = True,                 #따로 커밋 안 해도 됨.
        )

    def get_connection(self):
        conn = self.db_pool.connection()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        return conn, cur
    
    def close(self):
        self.db_pool.close()