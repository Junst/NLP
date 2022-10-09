https://github.com/Junst/KoNLPy-tTV


# KoNLPy_tTV
text To Video / 텍스트를 입력하면 그에 맞는 영상을 보여줍니다.

![image](https://github.com/Junst/KoNLPy_tTV/blob/master/GitPic/%EA%B7%B8%EB%A6%BC1.jpg)

## 개요
현 인공지능에서 가장 크게 대두되고 있는 NLP(자연어 처리)는 아직도 무궁무진하게 많은 과제를 남겨두고 있다. 그 중 가장 범용성이 뛰어난 활용 사례로는 “TEXT 입력을 통한 다양한 출력 프로그램”이 될 것이다. 즉, 입력 받은 text를 분석하여 그것에 맞는 text 혹은 Picture, Video, IoT 등 다양한 출력 형태의 결과물을 만들어내는 것이다.

본 프로젝트는 이러한 활용 사례를 참조하여 text To Video (T-TV)를 만들고자 한다. 여러 기술들이 조합하여 들어가겠지만, 가장 핵심이 되는 기술은 NLP(자연어 처리)의 영어 BERT이다. BERT는 자연어의 양방향적 해석, 문맥을 고려한 분석이 가능한, 현 NLP 기술에서 널리 쓰이고 있는 모델이다. 따라서 이 기술을 통해 입력 받은 text를 고려하여 그에 걸맞는 Video를 연출하는 프로젝트를 진행하고자 한다. 또한 Video를 구현하기 전, 먼저 Picture를 통해 결과를 도출하는 소규모 달성과제 역시 진행하고자 한다.
입력에서의 필요한 장비는 Jetson Nano와 Keyboard이며, 출력에서의 필요한 장비는 Monitor와 CUDA 등의 그래픽 카드로 예상하고 있다. 또한 기술적 한계가 크지 않다면, 마이크를 통한 음성 인식 역시 구현하고자 한다.

## 개발 환경, 언어 및 오픈소스 // Development Environment, Language and Opensource
JetsonNano : Ubuntu LXSession

Linux

Pycharm

Miniforge

Python 3.7.6

Tweepy 3.10.0

Selenium 4.0.0

Pip 18.1

KoNLPy 0.5.2

Jpype 1.2.0

Linux에서는 chromium을 사용했습니다.

## KoNLPy, SELENIUM
KoNLPy란, 한국어 정보처리를 위한 파이썬 패키지다. 자연어 처리(NLP)에서 형태소를 분리(형태소 단위 토크나이징)하는 데이터 전처리가 필요한데, 이 때 한국어 데이터 전처리를 할 때 사용하는 패키지이다.

텍스트를 형태소 단위로 분리하는 방법은 다음과 같다.

1.	단어 -> 품사 형태로 딕셔너리를 정의하고 이를 이용해 단어를 품사로 분리하는 방법
2.	딕셔너리를 사용하지 않고 모델을 통해 학습시키는 방법

KoNLPy는 이 중 1번의 방법을 사용한다. KoNLPy에는 총 5가지의 형태소 분석 방법을 제공한다. Hannanum, Kkma, Komoran, Mecab, Okt 5가지의 클래스를 제공한다. 각각의 장점과 단점이 있는데, 사전 로딩시간과 클래스 pos 메소드 실행 시간, 정규화 및 인터넷 텍스트, 비표준어 처리 능력 등을 모두 비교해본 결과 Okt가 본 프로젝트에 가장 적합하다. 따라서 본 프로젝트는 Okt를 적용하여 진행한다.



## 프로세스
![image](https://github.com/Junst/KoNLPy_tTV/blob/master/GitPic/%EA%B7%B8%EB%A6%BC2.png)

1. KeyBoard(와 Microphone)을 통해 Text Input을 받습니다.

2. 해당 Input을 Pytorch (NLP)로 분석합니다. : KoNLPy 

3. 분석된 데이터의 주제를 구글에서 크롤링을 합니다. : Selenium

4. MoviePy라는 오픈소스를 통해 해당 사진을 영상으로 인코딩합니다.

5. Speaker와 Monitor로 결과물을 출력합니다.


## 진행 과정
2021.11.24

먼저 텍스트의 Input을 받아 KoNLPy Okt에 해당 텍스트를 분석합니다. 예시 문장은 다음으로 설정하였다.

아라시가 좋아요. 아라시의 노래 중에 가장 좋아하는 노래는 아라시입니다. 아라시의 아라시, 제목과 노래가 똑같아욬ㅋㅋㅋ

KoNLPy에서 Noun_list 생성 및 배열을 통해 가장 빈도 수가 높은 단어를 체크합니다.
![image](https://github.com/Junst/KoNLPy_tTV/blob/master/GitPic/tTV_NLP1.png)

빈도 수가 높은 단어 2개를 단순히 합쳤습니다.
![image](https://github.com/Junst/KoNLPy_tTV/blob/master/GitPic/tTV_NLP2.PNG)

2021.11.30

입력한 문장의 형태소에 모든 명사를 가져오고, 그 명사들 중에 가장 빈도수가 높은 단어 두 개를 추출하였다. 이를 Selenium을 통해 검색한 후 이미지 크롤링을 한 결과 다음과 같은 결과를 얻었다.

![image](https://github.com/Junst/KoNLPy-tTV/blob/master/GitPic/%EA%B7%B8%EB%A6%BC1.png)

(중략)

![image](https://github.com/Junst/KoNLPy-tTV/blob/master/GitPic/pic2.png)

또한 “아라시 노래”라는 폴더에 Date “2021” 폴더가 생성되어, 크롤링한 사진을 모두 그곳에 저장했다.

![image](https://github.com/Junst/KoNLPy-tTV/blob/master/GitPic/%EA%B7%B8%EB%A6%BC3.png)

2021.12.02

크롤링 및 저장까지 완료된 후, 코드에 의해 사진을 Resize하여 영상으로 만들어준다. 해당 결과 이후에 폴더에 “mygeneratedvideo” 파일이 생성된다.

![image](https://github.com/Junst/KoNLPy-tTV/blob/master/GitPic/%EA%B7%B8%EB%A6%BC4.png)
![image](https://github.com/Junst/KoNLPy-tTV/blob/master/GitPic/%EA%B7%B8%EB%A6%BC5.png)

## 결과 // Conclusion & Discussion 
![image](https://github.com/Junst/KoNLPy-tTV/blob/master/result.gif)

본 프로젝트에서 처음 설정했던 목표인 “단어를 입력하여, 그에 맞는 사진을 가져와 영상을 제작하는 프로그램 제작”에 성공했다. 해당 프로젝트를 통해 간단한 영상 제작 및 사진 등의 조합을 기대할 수 있다. 

본 프로젝트에서 나온 결과물은 다음과 같이 응용할 수 있다. 첫째는 해당 영상의 사이즈를 조정하거나,fadein, fadeout과 같은 효과를 통해 동영상의 퀄리티를 변경할 수 있다. 다음으로 해당 크롤링을 사진이 아닌 영상으로 변환하여 영상과의 짜집기를 통해 교차편집 영상과 같은 고수준의 영상처리를 실행할 수 있다. 마지막으로, 텍스트로 인식 받은 KoNLPy를 활용하여 음성 인식 기술을 넣어 더욱 편리하고 짧은 프로세스의 영상 처리 프로그램으로 응용해볼 수 있다.
