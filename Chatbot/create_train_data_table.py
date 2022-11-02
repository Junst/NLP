import pymysql
from config.DatabaseConfig import * # DB 접속 정보 불러오기

db = None
try :
    db = pymysql.connect(
        host=DB_HOST,
        user = DB_USER,
        passwd = DB_PASSWORD,
        db= DB_NAME,
        port= DB_PORT,
        charset = 'utf8' # 'NoneType' object has no attribute 'encoding' 오류 'utf-8' -> 'utf8'
    )

    # 테이블 생성 sql 정의
    sql = '''
        CREATE TABLE IF NOT EXISTS `chatbot_train_data`(
        `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
        `intent` VARCHAR(45) NULL,
        `ner` VARCHAR(1024) NULL,
        `query` TEXT NULL,
        `answer` TEXT NOT NULL,
        `answer_image` VARCHAR(2048) NULL,
        PRIMARY KEY(`id`))
    ENGINE = InnoDB DEFAULT CHARSET = utf8
    '''
    # You have an error in your SQL syntax; 오류 발생 시 '를 `(탭키 위 물결표키)키로 해보기

    # 테이블 생성
    with db.cursor() as cursor :
        cursor.execute(sql)

except Exception as e:
    print(e)

finally:
    if db is not None :
        db.close()