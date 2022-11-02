from utils.Preprocess import Preprocess

sent = "내일 오전 10시에 탕수육 주문하고 싶어"
sent1 = "내일은 11월 3일, 아라시 데뷔 기념일이야"

# 전처리 객체 생성
p= Preprocess(userdic='../utils/user_dic.tsv')

# 형태소 분석기 실행
pos=p.pos(sent1)

# 품사 태그와 같이 키워드 출력
ret = p.get_keywords(pos, without_tag=False)
print(ret)

# 품사 태그 없이 키워드 출력
ret = p.get_keywords(pos, without_tag=True)
print(ret)