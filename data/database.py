import MySQLdb
from config.settings import DB_CONFIG


def init_connection():
    """데이터베이스 연결 초기화"""
    return MySQLdb.connect(
        user=DB_CONFIG["USER"],
        password=DB_CONFIG["PASSWORD"],
        host=DB_CONFIG["HOST"],
        port=DB_CONFIG["PORT"],
        db=DB_CONFIG["NAME"]
    )


def insert_restaurant_info(conn, restaurant_data):
    """음식점 정보 DB에 대량 삽입"""
    sql = """
        INSERT INTO content_restaurant_info 
        (id, name, x, y, road_address_name, url, type)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    values = [
        (
            data['id'],
            data['place_name'],
            data['x'],
            data['y'],
            data['road_address_name'],
            data['place_url'],
            data['category_name'].split('>')[-1].strip()
        )
        for data in restaurant_data
    ]

    cursor = conn.cursor()
    try:
        cursor.executemany(sql, values)
        conn.commit()
    finally:
        cursor.close()

def insert_restaurant_review(conn, restaurant_data):
    """음식점 리뷰 DB에 대량 삽입"""
    sql = """
        INSERT INTO content_restaurant_review 
        (restaurant_id, review, score)
        VALUES (%s, %s, %s)
    """

    values = [
        (
            data.get('restaurant_id'),  # 음식점 ID
            data.get('content', ''),  # 리뷰 내용
            data.get('rating', 0)  # 리뷰 점수
        )
        for data in restaurant_data
    ]

    cursor = conn.cursor()
    try:
        cursor.executemany(sql, values)
        conn.commit()
    finally:
        cursor.close()

def insert_good_words(conn, words):
    """긍정 단어 DB 대량 삽입"""
    sql = "INSERT INTO content_good_word (word) VALUES (%s)"
    cursor = conn.cursor()
    try:
        values = [(word,) for word in words]  # 튜플 형태로 변환
        cursor.executemany(sql, values)
        conn.commit()
    finally:
        cursor.close()


def insert_bad_words(conn, words):
    """부정 단어 DB 대량 삽입"""
    sql = "INSERT INTO content_bad_word (word) VALUES (%s)"
    cursor = conn.cursor()
    try:
        values = [(word,) for word in words]
        cursor.executemany(sql, values)
        conn.commit()
    finally:
        cursor.close()

def get_all_restaurant_info(conn):
    """모든 음식점 ID와 URL 가져오기"""
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id, url FROM content_restaurant_info")
        # 각 행을 딕셔너리 형태로 반환
        restaurant_list = [
            {"id": row[0], "url": row[1]}
            for row in cursor.fetchall()
        ]
        return restaurant_list
    finally:
        cursor.close()

def get_all_reviews(conn):
    """모든 리뷰 정보 가져오기"""
    cursor = conn.cursor()
    try:
        # 모든 컬럼 가져오기
        cursor.execute("SELECT * FROM content_restaurant_review")
        reviews = cursor.fetchall()
        # 컬럼 이름 포함해서 dict로 반환하려면
        columns = [desc[0] for desc in cursor.description]
        reviews_dict = [dict(zip(columns, row)) for row in reviews]
        return reviews_dict
    finally:
        cursor.close()

def get_all_scores(conn):
    """모든 점수 가져오기"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT score FROM content_restaurant_review")
        scores = [item[0] for item in cursor.fetchall()]
        return scores
    finally:
        cursor.close()


def get_restaurants_by_type(conn, food_type):
    """음식 종류로 음식점 검색"""
    cursor = conn.cursor()
    try:
        sql = """
        SELECT id, x, y 
        FROM content_restaurant_info 
        WHERE name LIKE %s OR type LIKE %s
        """
        cursor.execute(sql, (f'%{food_type}%', f'%{food_type}%'))
        return cursor.fetchall()
    finally:
        cursor.close()


def get_restaurant_details(conn, restaurant_id):
    """특정 음식점 상세 정보 가져오기"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT name, type, y, x, url FROM content_restaurant_info WHERE id = %s",
            [restaurant_id]
        )
        return cursor.fetchone()
    finally:
        cursor.close()


def get_restaurant_reviews(conn, restaurant_id):
    """특정 음식점 리뷰 가져오기"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT review FROM content_restaurant_review WHERE restaurant_id = %s",
            [restaurant_id]
        )
        return [item[0] for item in cursor.fetchall()]
    finally:
        cursor.close()


def get_restaurant_scores(conn, restaurant_id):
    """특정 음식점 점수 가져오기"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT score FROM content_restaurant_review WHERE restaurant_id = %s",
            [restaurant_id]
        )
        return [item[0] for item in cursor.fetchall()]
    finally:
        cursor.close()


def get_good_words(conn):
    """긍정 단어 목록 가져오기"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT word FROM content_good_word")
        return [item[0] for item in cursor.fetchall()]
    finally:
        cursor.close()


def get_bad_words(conn):
    """부정 단어 목록 가져오기"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT word FROM content_bad_word")
        return [item[0] for item in cursor.fetchall()]
    finally:
        cursor.close()


def init_db(conn):
    """테이블 생성 (존재하면 무시)"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_restaurant_info (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255),
            x DOUBLE,
            y DOUBLE,
            road_address_name VARCHAR(255),
            url VARCHAR(255),
            type VARCHAR(255)
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_restaurant_review (
            id INT AUTO_INCREMENT PRIMARY KEY,
            restaurant_id VARCHAR(255),
            review TEXT,
            score INT,
            FOREIGN KEY (restaurant_id) REFERENCES content_restaurant_info(id)
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_good_word (
            id INT AUTO_INCREMENT PRIMARY KEY,
            word VARCHAR(255) UNIQUE
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_bad_word (
            id INT AUTO_INCREMENT PRIMARY KEY,
            word VARCHAR(255) UNIQUE
        )
        """)
        conn.commit()
    finally:
        cursor.close()
