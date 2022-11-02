from konlpy.tag import Komoran
import pickle

class Preprocess:
    def __init__(self, word2index_dic='',userdic=None): # 생성자
        # 단어 인덱스 사전 불러오기
        if(word2index_dic != ''):
            f= open(word2index_dic,"rb")
            self.word_index = pickle.load(f)
            f.close()

        else:
            self.word_index = None


        # 형태소 분석기 초기화
        # 클래스가 생성될 때 형태소 분석기 인스턴스 객체를 생성한다.
        # userdic 인자에는 사용자 정의 사전 파일의 경로를 입력할 수 있다.
        self.komoran = Komoran(userdic=userdic)

        # 제외할 품사
        # 참조 : https://docs.komoran.kr/firststep/postypes.html
        # 관계언 제거, 기호 제거
        # 어미 제거
        # 접미사 제거

        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

    # 형태소 분석기 POS 태거
    # 클래스 외부에서는 코모란 형태소 분석기 객체를 직접적으로 호출할 일이 없게 하기 위해 정의한 래퍼 함수이다.
    # 형태소 분석기 종류를 바꾸게 될 경우 이 래퍼 함수 내용만 변경하면 되므로 유지보수 측면에서 장점이 많다.
    def pos(self, sentence):
        return self.komoran.pos(sentence)

    # 불용어 제거 후 필요한 품사 정보만 가져오기
    # 생성자에서 정의한 self.exclusion_tags 리스트에서 해당하지 않는 품사 정보만 키워드로 저장한다.
    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags
        word_list=[]
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
        return word_list

    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []
        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                # 해당 단어가 사전에 없는 경우 OOV 처리
                w2i.append(self.word_index['OOV'])
        return w2i
    