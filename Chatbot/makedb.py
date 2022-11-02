# https://curriculum.cosadama.com/basic-sql/4-2/

# 1. 실행하기
import pymysql

# 2. connect
conn = pymysql.connect(host='root', port=3306, user = 'root', password='junstyle12', charset = 'utf8')

# 3. cursor 사용하기
cursor = conn.cursor()

# 4. sql 구문 (DB 만들어주기) 만들기
sql = 'CREATE DATABASE CHATBOT'

# 5. execute
cursor.execute(sql)

# 6. commit (최종 변경)
conn.commit()

# 7. DB 닫아주기
conn.close()