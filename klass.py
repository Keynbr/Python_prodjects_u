import random
class Coder:
    def __init__(self,key):
        self.xN()
        pass

    @staticmethod
    def forward_premutation(message):
        order = (58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8, 57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3, 61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7)
        new_message = (message[val-1] for val in order)
        return tuple(new_message)

    @staticmethod
    def inverse_premutation(message):
        order = (40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,34,2,42,10,50,18,58,26,33,1,41,9,49,17,57,25)
        new_message = (message[val-1] for val in order)
        return tuple(new_message)

    def xN(self):
        self._x = 3571
        self._N =  8550052947598282327


    def generateShakeMessage(self):
        m = 1
        self._s = random.randint(10000, 1000000)
        for i in range(self._s):
            m = (m * self._x) % self._N
        return m

    @staticmethod
    def binary(i_val):
        a = [0 for j in range(56)]
        for i in range(56, -1, -1):
            a[6 - i] = i_val // 2 ** i
            if a[6 - i] == 1:
                i_val = i_val - 2 ** i
        return tuple(a)

    def setShakeKey(self, m2):
        for i in range(self._s):
            a = (m2 * self._x) % self._N
        self._key = Coder.binary(a % (2**56))

    @property
    def key(self):
        return self._key


    @key.setter
    def key(self, new_key):
        if len(new_key)== 56:
            self._key = new_key
            pass

    @staticmethod
    def split64(message):
        if len(message) == 64:
            return tuple((message,))
        elif len(message) < 64:
            a = 64 - len(message)
            new_message = message + tuple([1] * a)
            return tuple((new_message,))
        else:
            n = int(len(message) / 64)
            new_message2 = []
            for i in range(n):
                new_message = message[:64:]
                message = message[64::]
                new_message2.append(new_message)
            a = 64 - len(message)
            new_message2.append(message + tuple([1] * a))
            return tuple(new_message2)

    @staticmethod
    def cikl_sdvig(arr, i):
        while i > 0:
            arr = arr[1:] + arr[:1]
            i -= 1
        return arr

    #def keyGen(self,i):
    #    return tuple(1 * (v < i) for v in range(48))

    def keyGen(self,i):
        #self._key = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        a = self._key
        key_e = []
        T = []
        for j in range(8):
            key_e1 = a[:7:]
            a = a[7::]
            count = 0
            for k in range(7):
                count += int(key_e1[k])
            if count % 2 == 0:
                key_e += list(key_e1) + [1]
            else:
                key_e += list(key_e1) + [0]
        cc = (57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36)
        dd = (63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22, 14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4)
        c = tuple(key_e[val - 1] for val in cc)
        d = tuple(key_e[val - 1] for val in dd)
        sdvig = (1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1)
        T = Coder.cikl_sdvig(c,sdvig[i-1]) + Coder.cikl_sdvig(d,sdvig[i-1])
        order = (14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10, 23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2, 41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48, 44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32)
        k_i = tuple(T[val-1] for val in order)
        return k_i

    def f(self, in_, k_i):
        order = (32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17, 16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25, 24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1)
        in_e = tuple(in_[val-1] for val in order)
        e = Coder.my_xor(in_e, k_i)
        T = []
        for i in range(8):
            a = []
            b = []
            e1 = e[:6:]
            e = e[6::]
            a.append(e1[0]), a.append(e1[5])
            b.append(e1[1]), b.append(e1[2]), b.append(e1[3]), b.append(e1[4])
            s1 = [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                  [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                  [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                  [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]]
            s2 = [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                  [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                  [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                  [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]]
            s3 = [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
                  [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
                  [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
                  [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]]
            s4 = [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
                  [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
                  [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
                  [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]]
            s5 = [[2 ,12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
                  [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
                  [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
                  [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]]
            s6 = [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
                  [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
                  [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
                  [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]]
            s7 = [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
                  [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
                  [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
                  [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]]
            s8 = [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
                  [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
                  [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
                  [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]
            S = (s1,s2,s3,s4,s5,s6,s7,s8)
            si_ = S[i]
            bi_ = si_[int(a[0]*2+a[1])][int(b[0]*8+b[1]*4+b[2]*2+b[3])]
            T.append(bi_)
        T2 = []
        for val in T:
            yy = [0, 0, 0, 0]
            for j in range(4, -1, -1):
                yy[3 - j] = val // 2 ** j
                if yy[3 - j] == 1:
                    val = val - 2 ** j
            T2 += yy
        order1 = (16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10, 2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25)
        Res = tuple(T2[val -1] for val in order1)
        return tuple(Res)

    @staticmethod
    def my_xor(x, y):
        xy = []
        for i in range(len(x)):
            if x[i] == y[i]:
                xy.append(0)
            else:
                xy.append(1)
        return tuple(xy)

    def feistel(self, L, R, k_i):
        L2 = R
        R2 = Coder.my_xor(L,self.f(R,k_i))
        return L2, R2

    def inv_feistel(self, L2, R2, k_i):
        L = Coder.my_xor(R2,self.f(L2,k_i))
        R = L2
        return L, R

    def code(self, message):
        blok = Coder.split64(message)
        new_message = ()
        for x in blok:
            coded = Coder.forward_premutation(x)
            R = coded[int(len(coded) / 2)::]
            L = coded[:int(len(coded) / 2):]
            for i in range(1, 17):
                k_i = self.keyGen(i)
                L, R = self.feistel(L, R, k_i)
            coded2 = L + R
            new_message += Coder.inverse_premutation(coded2)
        return new_message

    def decode(self, coded):
        blok = Coder.split64(coded)
        message = ()
        for x in blok:
            x = Coder.forward_premutation(x)
            R2 = x[int(len(x) / 2)::]
            L2 = x[:int(len(x) / 2):]
            for i in range(1, 17):
                k_i = self.keyGen(17-i)
                L2, R2 = self.inv_feistel(L2, R2, k_i)
            x = L2 + R2
            message += Coder.inverse_premutation(x)
        return message