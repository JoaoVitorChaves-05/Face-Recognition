import cv2
import face_recognition as fr
import os
import time

def exec(begin, end, message):
    return print(message, end - begin, "seconds")

def load_face_encodings(dirs):
    face_encodings = []
    count = 0

    for dir in dirs:
        print('./images/lfw/' + dir + '/' + os.listdir('./images/lfw/' + dir)[0])
        photo = fr.load_image_file('./images/lfw/' + dir + '/' + os.listdir('./images/lfw/' + dir)[0])
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        try:
            encode_photo = fr.face_encodings(photo)[0]
            face_encodings.append(encode_photo)
        except:
            pass
        if count == 20:
            break
        else:
            count += 1

    return face_encodings

def load_face_encoding_to_compare(file):
    photo = fr.load_image_file(file)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    encode_photo = fr.face_encodings(photo)[0]
    return encode_photo

def compare_encodings(user_encoding, encodings):
    for encoding in encodings:
        compare = fr.compare_faces([user_encoding], encodings)
        if compare[0] == True:
            return print('You are authorized')
    return print('You are not authorized')



inicio = time.time()
loaded_face_encondings = load_face_encodings(os.listdir('./images/lfw/'))
fim = time.time()
exec(inicio, fim, "Execution time to load the face encodings:")

face_to_compare = input("Insert the filename to compare: ")
inicio_2 = time.time()
face_encoding_to_compare = load_face_encoding_to_compare('./compare/' + face_to_compare)
fim_2 = time.time()
exec(inicio_2, fim_2, "Execution time to load the face encoding to compare:")

inicio_3 = time.time()
compare_encodings(face_encoding_to_compare, loaded_face_encondings)
fim_3 = time.time()
exec(inicio_3, fim_3, "Execution time to compare faces")

'''
my_photo_1 = fr.load_image_file('./images/my_photo_1.png')
my_photo_1 = cv2.cvtColor(my_photo_1, cv2.COLOR_BGR2RGB)

my_photo_2 = fr.load_image_file('./images/eu.png')
my_photo_2 = cv2.cvtColor(my_photo_2, cv2.COLOR_BGR2RGB)

#faceLoc = fr.face_locations(my_photo_1)[0]
#print(faceLoc)
#cv2.rectangle(my_photo_1, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

encode_my_photo_1 = fr.face_encodings(my_photo_1)[0]
encode_my_photo_2 = fr.face_encodings(my_photo_2)[0]

compare = fr.compare_faces([encode_my_photo_1], encode_my_photo_2)

print(compare)
'''

