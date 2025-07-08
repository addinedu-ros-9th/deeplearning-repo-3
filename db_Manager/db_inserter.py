import json

class ObjectTypeInserter:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def insert_object_type(self, name: str) -> int:
        conn, cur = self.db_connector.get_connection()
        try:
            # 이미 있는지 확인
            cur.execute(
                """
                SELECT id FROM object_type WHERE name = %s
                """, (name,)
                )
            result = cur.fetchone()   # 위의 실행 결과의 row 반환 ex) {'id' : 3} 또는 None
            if result:
                return result['id'] # id 값 반환
            else:
                # 없으면 새로 추가
                cur.execute(
                    """
                    INSERT INTO object_type (name) VALUES (%s)
                    """, (name,)
                    )
                return cur.lastrowid # 방금 삽입된 행의 id 값을 반환
        except Exception as e:
            print(f"[DB ERROR] {e}")
            raise
        finally:
            cur.close()
            conn.close()


class DriveSessionInserter:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def insert_drive_session(self, start_time, end_time, total_distance):
        conn, cur = self.db_connector.get_connection()

        try:
            cur.execute(
                """
                INSERT INTO drive_session (start_time, end_time, total_distance)
                VALUES (%s, %s, %s)
                """, (start_time, end_time, total_distance)
            )
            return cur.lastrowid
        except Exception as e:
            print(f"[DB ERROR] {e}")
            raise
        finally:
            cur.close()
            conn.close()


class ActionTypeInserter:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def insert_action_type(self, name: str) -> int:
        conn, cur = self.db_connector.get_connection()
        try:
            # 같은 name이 이미 있는지 조회
            cur.execute(
                """
                SELECT id FROM action_type WHERE name = %s
                """, (name,)
                )
            result = cur.fetchone()
            if result:
                return result['id']
            else:
                # 없으면 새로 INSERT
                cur.execute(
                    """
                    INSERT INTO action_type (name) VALUES (%s)
                    """, (name,)
                    )
                return cur.lastrowid
        except Exception as e:
            print(f"[DB ERROR] {e}")
            raise
        finally:
            cur.close()
            conn.close()


class DetectedObjectInserter:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def insert_detected_object(self,
                                session_id: int,
                                object_type_id: int,
                                detected_time,   # datetime 객체
                                confidence: float,
                                bbox: dict,
                                position: float):
        conn, cur = self.db_connector.get_connection()
        try:
            cur.execute(
                """
                INSERT INTO detected_object (session_id, object_type_id, detected_time, confidence, bbox, position)
                VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                session_id,
                object_type_id,
                detected_time,
                confidence,
                json.dumps(bbox),     # dict → JSON 문자열
                json.dumps(position)  # dict → JSON 문자열
            ))
            return cur.lastrowid
        except Exception as e:
            print(f"[DB ERROR] {e}")
            raise
        finally:
            cur.close()
            conn.close()


class ActionLogInserter:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def insert_action_log(self,
                          object_id: int,
                          action_type_id: int,
                          performed_time,  # datetime 객체
                          delay: float):
        conn, cur = self.db_connector.get_connection()
        try:
            cur.execute(
                """
                INSERT INTO action_log (object_id, action_type_id, performed_time, delay)
                VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE 
                performed_time = VALUES(performed_time), delay = VALUES(delay)
                """, (
                object_id,
                action_type_id,
                performed_time,
                delay,
            ))
            return cur.lastrowid
        except Exception as e:
            print(f"[DB ERROR] {e}")
            raise
        finally:
            cur.close()
            conn.close()

class DriveSessionUpdater:
    def __init__(self, db_connector):
        self.db_connector = db_connector

    def update_end_time_and_distance(self, session_id: int, end_time, total_distance: float):
        conn, cur = self.db_connector.get_connection()
        try:
            cur.execute(
                """
                UPDATE drive_session
                SET end_time = %s, total_distance = %s
                WHERE id = %s
                """, (end_time, total_distance, session_id)
            )
        except Exception as e:
            print(f"[DB ERROR] {e}")
            raise
        finally:
            cur.close()
            conn.close()