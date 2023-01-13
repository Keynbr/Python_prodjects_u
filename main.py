from klass import Coder

alphavit = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ .,:"!?'

def binary(i_val):
    a = [0, 0, 0, 0, 0, 0, 0]
    for i in range(6, -1, -1):
        a[6-i] = i_val // 2**i
        if a[6-i] == 1:
            i_val = i_val - 2**i
    return tuple(a)

d = {}
for idx, ch in enumerate(alphavit):
    d[ch] = binary(idx)
#print(d)

d2 = {}
for key in d:
    val = d[key]
    d2[val] = key
#print(d2)

def message2Binary(message):
    result = ()
    for ch in message:
        result = result + d[ch]
    result = result + (1, 1, 1, 1, 1, 1, 1)
    return result

def binary2Message(bin):
    a = 0
    b = 7
    result = ''
    while bin[a:b:] != (1, 1, 1, 1, 1, 1, 1):
        result = result + d2[bin[a:b:]]
        a = b
        b += 7
    return result

coder = Coder(0)
x = coder.generateShakeMessage()
coder.setShakeKey(x)
message = 'Ура'
bin = message2Binary(message)
coded = coder.code(bin)
bin2 = coder.decode(coded)
message2 = binary2Message(bin2)
print(message2)

coder = Coder(0)
x = coder.generateShakeMessage()
coder.setShakeKey(x)
message = 'Наконец то оно работает'
bin = message2Binary(message)
coded = coder.code(bin)
bin2 = coder.decode(coded)
message2 = binary2Message(bin2)
print(message2)

coder = Coder(0)
x = coder.generateShakeMessage()
print(x)
x = input()
coder.setShakeKey(int(x))
message = 'Алексей Михайлович, слишком сложная работа'
bin = message2Binary(message)
coded = coder.code(bin)
a = ''
for i in range(len(coded)):
    a += str(coded[i])
print(a)
b = input()
bin2 = coder.decode(tuple(int(bit) for bit in b))
message2 = binary2Message(bin2)
print(message2)


# Создаём два кодера для проверки взаимодействия
coder1 = Coder(1)
coder2 = Coder(2)

# Каждый кодер генерирует секретное слово и сообщение для
# рукопожатия
m1 = coder1.generateShakeMessage()
m2 = coder2.generateShakeMessage()

# Сообщения должны быть разными
print(f'Сообщение для рукопожатия 1 кодера: {m1}')
print(f'Сообщение для рукопожатия 2 кодера: {m2}')

# Обмен сообщениями для рукопожатия и генерирование ключа
coder1.setShakeKey(m2)
coder2.setShakeKey(m1)

# Ключи должны быть одинаковыми
# Тут key – это @property, если у вас функция, надо добавить скобки
print(f'Ключ первого кодера: {coder1.key}')
print(f'Ключ второго кодера: {coder2.key}')

# Сообщение для обмена
message ='Привет, друзья.'
print(message)
# Преобразование в двоичную форму
bin_message = message2Binary(message)
print(bin_message)
# Первый кодер кодирует сообщение
coded_message = coder1.code(bin_message)
print (coded_message)
# Второй кодер декодирует сообщение
decoded_message = coder2.decode(coded_message)
print(decoded_message)
# Восстановление сообщения в текстовый вид
message2 = binary2Message(decoded_message)
print(message2)