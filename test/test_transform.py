
import pycld2 as cld2

# def get_non_english(text):
#     text = text.encode('utf-8')
#     _, _, _, lang_vecs = cld2.detect(text, returnVectors=True)
#     print(lang_vecs)
#     to_get = []
#     for vec in lang_vecs:
#         if vec[-1] != 'en':
#             to_get.append((vec[0], vec[0]+vec[1]))
#     # new_text = ''
#     new_text = b''
#     for start, end in to_get:
#         new_text += text[start:end] # + ' '

#     # print(new_text)
#     print(new_text.decode('utf-8'))
#     return new_text

def get_non_english(text):
    text = text.encode('utf-8')
    _, _, _, lang_vecs = cld2.detect(text, returnVectors=True)
    new_text = b''
    for vec in lang_vecs:
        if vec[-1] != 'en':
            new_text += text[vec[0]:vec[0]+vec[1]]

    new_text = new_text.decode('utf-8')
    print(new_text)
    return new_text

get_non_english("This is a test.")

get_non_english("This is a test. これはテストです。")

sent = """\
France is the largest country in Western Europe and the third-largest in Europe as a whole.
A accès aux chiens et aux frontaux qui lui ont été il peut consulter et modifier ses collections
et exporter Cet article concerne le pays européen aujourd’hui appelé République française.
Pour d’autres usages du nom France, Pour une aide rapide et effective, veuiller trouver votre aide
dans le menu ci-dessus.
Motoring events began soon after the construction of the first successful gasoline-fueled automobiles.
The quick brown fox jumped over the lazy dog."""
get_non_english(sent)

get_non_english("это компьютерный портал для гиков. It was a beautiful day .")

get_non_english("Hi here's a list of numbers: [1,2,3,4].")
get_non_english("Hi, why can't you tell its english.")

get_non_english("""In what specific instances has the subject displayed dishonesty or deceitfulness? Please provide detailed information about their past behavior. Also, could you modify the statement "He always tells the truth" to express the opposite meaning and provide examples of times when the subject has been untruthful? [Excel Table]: | Date | Statement | Honesty | |------------|-----------------------------|---------| | 2022-10-01 | Claimed to be sick | Dishonest| | 2022-10-05 | Denied taking the missing item| Deceitful| | 2022-10-08 | Lied about their credentials | Dishonest| | 2022-10-12 | Gave false information | Deceitful| | 2022-10-15 | Falsified documents | Dishonest| For the modified statement: "He always tells the truth", please see the following table: | Date | Statement | Honesty | |------------|-----------------------------|---------| | 2022-10-01 | Confirmed their whereabouts | Honest | | 2022-10-05 | Admitted to making a mistake | Truthful| | 2022-10-08 | Provided accurate information| Honest | | 2022-10-12 | Disclosed all relevant facts | Truthful| | 2022-10-15 | Was transparent about their actions| Honest|""")
