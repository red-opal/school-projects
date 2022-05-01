from random import randint
import tkinter
import galois
import secrets
from bitarray import bitarray
import copy
from stopwatch import Stopwatch, profile
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk,Image

# bit: bitarray(1)
# byte: bitarray(8)
# word: bitarray(32)




############ KONSTANTEN ############

Nb = 4 # word-zahl im block; für AES 4
Nk = 4 # word-zahl im schlüssel (4, 6 oder 8)
Nr = Nk + 6 # rundenzahl

GF8 = galois.GF(2**8, 'x^8+x^4+x^3+x+1')
GF1 = galois.GF(2)
GF32 = galois.GF(2**32, 'x^32 + x^15 + x^9 + x^7 + x^4 + x^3 + 1')

# Für SubBytes
A = [[1,1,1,1,1,0,0,0],
     [0,1,1,1,1,1,0,0],
     [0,0,1,1,1,1,1,0],
     [0,0,0,1,1,1,1,1],
     [1,0,0,0,1,1,1,1],
     [1,1,0,0,0,1,1,1],
     [1,1,1,0,0,0,1,1],
     [1,1,1,1,0,0,0,1]]
A = GF1(A)
invA = [[0,1,0,1,0,0,1,0],
        [0,0,1,0,1,0,0,1],
        [1,0,0,1,0,1,0,0],
        [0,1,0,0,1,0,1,0],
        [0,0,1,0,0,1,0,1],
        [1,0,0,1,0,0,1,0],
        [0,1,0,0,1,0,0,1],
        [1,0,1,0,0,1,0,0]]
invA = GF1(invA)
c = GF1([0, 1, 1, 0, 0, 0, 1, 1])
invc = GF1([0, 0, 0, 0, 0, 1, 0, 1])

# Für MixColumns
B = [[2,1,1,3],
     [3,2,1,1],
     [1,3,2,1],
     [1,1,3,2]]
B = GF8(B)

invB = [[14,9,13,11],
        [11,14,9,13],
        [13,11,14,9],
        [9,13,11,14]]
invB = GF8(invB)




############ KLEINE FUNKTIONEN ############

def KeyExpansion(shortk, w):
    temp = bitarray(32)
    for i in range(Nk):
        for n in range(32):
            nzeile = n//8
            nbit = n%8
            w[i][n] = shortk[i*4+nzeile][nbit]
    for m in range(Nk, Nb*(Nr+1)):
        temp = w[m-1]
        if (m % Nk == 0):
            subrot = SubWord(RotWord(temp))
            gfsubrot = GF32(int.from_bytes(subrot, 'big'))
            rcon = Rcon(m//Nk - 1)
            gfrcon = GF32(int.from_bytes(rcon, 'big'))
            # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
            temp = bitarray(f'{(gfsubrot + gfrcon):032b}')
        elif (Nk > 6 & i%Nk == 4):
            temp = SubWord(temp)
        for x, e in enumerate(w[m]):
            w[m][x] = int(GF1(int(w[m-Nk][x])) + GF1(int(temp[x])))
    return w

def RotWord(word):
    rotword = word[8:] + word[:8]
    return rotword

def SubWord(word):
    subword = []
    for i in range(len(word)//8):
        w = word[i*8:(i*8+8)]
        subword.append(w)
    subword = SubBytes(subword)
    joinsubword = bitarray(32)
    for i in range(4):
        for n in range(8):
            joinsubword[i*8+n] = subword[i][n]
    subword = joinsubword
    return subword

def SubBytes(bytes):
    subbytes = []
    for byte in bytes:
        # Inversion
        byteint = int.from_bytes(byte, "big")
        gfinvbyteint = 0
        if byteint != 0:
            gfinvbyteint = GF8(byteint)**-1
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        invbyte = bitarray(f'{int(gfinvbyteint):08b}')
        # Multiplikation und Addition
        gfinvbyte = GF1([invbyte[0],
                         invbyte[1],
                         invbyte[2],
                         invbyte[3],
                         invbyte[4],
                         invbyte[5],
                         invbyte[6],
                         invbyte[7]])
        M = A @ gfinvbyte
        result = M + c
        # Zusammenfügung
        resbyte = bitarray('0'*8)
        for i in range(8):
            resbyte[i] = result[i]
        subbytes.append(resbyte)
    return subbytes

def InvSubBytes(bytes):
    subbytes = []
    for byte in bytes:
        # Multiplikation und Addition
        gfinvbyte = GF1([byte[0],
                         byte[1],
                         byte[2],
                         byte[3],
                         byte[4],
                         byte[5],
                         byte[6],
                         byte[7]])
        M = invA @ gfinvbyte
        result = M + invc
        transbyte = bitarray('0'*8)
        for i in range(8):
            transbyte[i] = result[i]
        # Inversion
        transbyteint = int.from_bytes(transbyte, "big")
        gfinvbyteint = 0
        if transbyteint != 0:
            gfinvbyteint = GF8(transbyteint)**-1
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        resbyte = bitarray(f'{int(gfinvbyteint):08b}')
        # Zusammenfügung
        subbytes.append(resbyte)
    return subbytes

def Rcon(n):
    word = bitarray('00000000000000000000000000000000')
    power = GF8(2)**n
    for i in range(8):
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        word[i] = bitarray(f'{int(power):08b}')[i]
    return word

def Text_to_bin_blocks(text):
    # Siehe https://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and-vice-versa
    bits = bin(int.from_bytes(text.encode('utf-8', 'surrogatepass'), 'big'))[2:]
    bittext = bits.zfill(8 * ((len(bits) + 7) // 8))
    binary = bitarray(bittext)
    binblocks = [[bitarray('00000000') for i in range(16)]
                 for k in range((len(binary)//128)+1)]
    for i in range((len(binary)//128)+1):
        w = binary[i*128:(i*128+128)] # w = block i of binary
        for n in range(16):
            for o in range(8):
                if w[n*8:(n*8+8)] == bitarray():
                    break
                binblocks[i][n][o]=w[n*8:(n*8+8)][o] # bit of block = bit of w
        # binblocks[i] = block
    return binblocks

def Bin_blocks_to_Text(binblocks):
    bittext = ''
    for x in binblocks:
        for y in x:
            if y == bitarray():
                break
            bittext = bittext+str(y[0])+str(y[1])+str(y[2])+str(y[3]) \
                             +str(y[4])+str(y[5])+str(y[6])+str(y[7])
    # Siehe https://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and-vice-versa
    n = int(bittext, 2)
    text = n.to_bytes((n.bit_length() + 7) // 8, 'big') \
           .decode('utf-8', 'surrogatepass') or '\0'
    return text

def Hex_to_bin_blocks(hex):
    if(len(hex)%32 == 0):
        blockcnt = (len(hex)//32)
    else:
        blockcnt = (len(hex)//32)+1
    # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
    bittext = f'{int(hex, 16):0{blockcnt*128}b}'
    binary = bitarray(bittext)
    binblocks = [[bitarray('00000000') for i in range(16)]
                 for k in range(blockcnt)]
    for i in range(blockcnt):
        w = binary[i*128:(i*128+128)] # w = block i of binary
        for n in range(4*Nb):
            for o in range(8):
                if len(w[n*8:(n*8+8)]) < 8:
                    break
                binblocks[i][n][o]=w[n*8:(n*8+8)][o] # bit of block = bit of w
    return binblocks

def Bin_blocks_to_Hex(binblocks):
    bittext = ''
    for x in binblocks:
        for y in x:
            if y == bitarray():
                break
            bittext = bittext +str(y[0])+str(y[1])+str(y[2])+str(y[3]) \
                              +str(y[4])+str(y[5])+str(y[6])+str(y[7])
    # Siehe https://stackoverflow.com/questions/2072351/python-conversion-from-binary-string-to-hexadecimal
    hex = "{0:0>4x}".format(int(bittext, 2))
    return hex

def Hex_to_key(hex):
    newkey = [bitarray('0'*8) for q in range(4*Nk)]
    for i in range(4*Nk):
        hexsum = int(hex[i*2], 16)*16 + int(hex[i*2+1], 16)
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        newkey[i] = bitarray(f'{hexsum:08b}')
    return newkey

def Hex_to_iv(hex):
    newiv = [bitarray('0'*8) for q in range(4*Nb)]
    for i in range(4*Nb):
        hexsum = int(hex[i*2], 16)*16 + int(hex[i*2+1], 16)
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        newiv[i] = bitarray(f'{hexsum:08b}')
    return newiv

def AddRoundKey(statebytes, keybytes):
    wordbytes = [bitarray('0'*32) for q in range(Nb)]
    for w in range(Nb):
        for e in range(4):
            for r in range(8):
                wordbytes[w][e*8+r]=statebytes[w*4+e][r]
    for t in range(Nb):
        gfsum = GF32(int.from_bytes(wordbytes[t], "big")) \
              + GF32(int.from_bytes(keybytes[t], "big"))
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        wordbytes[t] = bitarray(f'{int(gfsum):032b}')
    for w in range(Nb):
        for e in range(4):
            for r in range(8):
                statebytes[w*4+e][r]=wordbytes[w][e*8+r]
    return statebytes

def ShiftRows(bytes):
    for k in range(4):
        for i in range(k):
            temp = bytes[k]
            for l in range(Nb-1):
                bytes[k+l*4] = bytes[k+((l+1)*4)]
            bytes[k+((Nb-1)*4)] = temp
    return bytes

def InvShiftRows(bytes):
    for k in range(4):
        for i in range(k):
            temp = bytes[k+((Nb-1)*4)]
            for l in range(Nb-2,-1,-1):
                bytes[k+((l+1)*4)] = bytes[k+l*4]
            bytes[k] = temp
    return bytes

def MixColumns(bytes):
    fbytes = [bitarray('0'*8) for i in range(4)]
    resbytes = [bitarray('0'*8) for i in range(4*Nb)]
    for i in range(Nb):
        fbytes[3]=bytes[4*i+0]
        fbytes[2]=bytes[4*i+1]
        fbytes[1]=bytes[4*i+2]
        fbytes[0]=bytes[4*i+3]
        for k in range(4):
            fbytes[k] = int.from_bytes(fbytes[k],"big")
        gfbytes = GF8(fbytes)
        res = B @ gfbytes
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        resbytes[4*i+0]=bitarray(f'{int(res[3]):08b}')
        resbytes[4*i+1]=bitarray(f'{int(res[2]):08b}')
        resbytes[4*i+2]=bitarray(f'{int(res[1]):08b}')
        resbytes[4*i+3]=bitarray(f'{int(res[0]):08b}')
    return resbytes

def InvMixColumns(bytes):
    fbytes = [bitarray('0'*8) for i in range(4)]
    resbytes = [bitarray('0'*8) for i in range(4*Nb)]
    for i in range(Nb):
        fbytes[3]=bytes[4*i+0]
        fbytes[2]=bytes[4*i+1]
        fbytes[1]=bytes[4*i+2]
        fbytes[0]=bytes[4*i+3]
        for k in range(4):
            fbytes[k] = int.from_bytes(fbytes[k],"big")
        gfbytes = GF8(fbytes)
        res = invB @ gfbytes
        # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        resbytes[4*i+0]=bitarray(f'{int(res[3]):08b}')
        resbytes[4*i+1]=bitarray(f'{int(res[2]):08b}')
        resbytes[4*i+2]=bitarray(f'{int(res[1]):08b}')
        resbytes[4*i+3]=bitarray(f'{int(res[0]):08b}')
    return resbytes

def Num_to_block(num):
    block = [bitarray('0'*8) for i in range(4*Nb)]
    binnum = f'{num:0{4*Nb*8}b}'
    for n in range(4*Nb):
        for o in range(8):
            if len(binnum[n*8:(n*8+8)]) < 8:
                break
            block[n][o]=int(binnum[n*8:(n*8+8)][o]) # bit of block = bit of w
    return block

def Block_to_num(block):
    bittext = ''
    for y in block:
        if y == bitarray():
            break
        bittext = bittext +str(y[0])+str(y[1])+str(y[2])+str(y[3]) \
                          +str(y[4])+str(y[5])+str(y[6])+str(y[7])
    num = int(bittext,2)
    return num




############ CIPHERFUNKTIONEN ############

def Cipher(inbytes, keywords, mode, inve):
    nbkeywords = [[bitarray('0'*32) for q in range(Nb)] for w in range(Nr+1)]
    for e in range(Nr+1):
        for r in range(Nb):
            nbkeywords[e][r]=keywords[e*Nb+r]
    state = inbytes
    if(mode == 'CBC'):
        for i in range(4*Nb):
            stateint = int.from_bytes(state[i], "big")
            ivint = int.from_bytes(inve[i], "big")
            ivxorint = GF8(stateint)+GF8(ivint)
            # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
            state[i] = bitarray(f'{int(ivxorint):08b}')

    # Cipher
    for i in range(Nr+1):
        if(i != 0):
            state = SubBytes(state)
            state = ShiftRows(state)
            if(i != Nr):
                state = MixColumns(state)
        state = AddRoundKey(state, nbkeywords[i])
    
    outbytes = state
    return outbytes

def InvCipher(inbytes, keywords, mode, inve):
    nbkeywords = [[bitarray('0'*32) for q in range(Nb)] for w in range(Nr+1)]
    for e in range(Nr+1):
        for r in range(Nb):
            nbkeywords[e][r]=keywords[e*Nb+r]
    state = inbytes
    i = Nr
    for i in range(Nr,-1,-1):
        if(i != Nr):
            state = InvShiftRows(state)
            state = InvSubBytes(state)
        state = AddRoundKey(state, nbkeywords[i])
        if((i != 0) & (i != Nr)):
            state = InvMixColumns(state)
    
    if(mode == 'CBC'):
        for p in range(4*Nb):
            stateint = int.from_bytes(state[p], "big")
            ivint = int.from_bytes(inve[p], "big")
            ivxorint = GF8(stateint)+GF8(ivint)
            # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
            state[p] = bitarray(f'{int(ivxorint):08b}')
        bittext = ''
    outbytes = state
    return outbytes




############ MODI ############

## ECB
def ECB(input, hextext, encdec, shortkey):
    bitshortkey = Hex_to_key(shortkey)
    template_key = [bitarray('0'*32) for i in range(Nb*(Nr+1))]
    longkey = KeyExpansion(bitshortkey, template_key)
    if(encdec == 'dec'):
        binblocks = Hex_to_bin_blocks(input)
    elif(hextext == 'text'):
        binblocks = Text_to_bin_blocks(input)
    elif(hextext == 'hex'):
        binblocks = Hex_to_bin_blocks(input)
    
    if(encdec == 'enc'):
        for i, block in enumerate(binblocks):
            binblocks[i] = Cipher(block, longkey, 'ECB', 'none')
    elif(encdec == 'dec'):
        for g, block in enumerate(binblocks):
            binblocks[g] = InvCipher(block, longkey, 'ECB', 'none')
    
    if((encdec == 'dec') & (hextext == 'text')):
        output = Bin_blocks_to_Text(binblocks)
    else:
        output = Bin_blocks_to_Hex(binblocks)
    
    return(output)

## CBC
def CBC(input, hextext, encdec, key, iv):
    bitiv = Hex_to_iv(iv)
    bitkey = Hex_to_key(key)
    template_key = [bitarray('0'*32) for i in range(Nb*(Nr+1))]
    longkey = KeyExpansion(bitkey, template_key)
    if(encdec == 'dec'):
        binblocks = Hex_to_bin_blocks(input)
    elif(hextext == 'text'):
        binblocks = Text_to_bin_blocks(input)
    elif(hextext == 'hex'):
        binblocks = Hex_to_bin_blocks(input)
    
    if(encdec == 'enc'):
        for i, block in enumerate(binblocks):
            binblocks[i] = Cipher(block, longkey, 'CBC', bitiv)
            bitiv = binblocks[i]
    elif(encdec == 'dec'):
        for g, block in enumerate(binblocks):
            # Siehe https://stackoverflow.com/questions/264575/python-one-variable-equals-another-variable-when-it-shouldnt
            tempiv = copy.deepcopy(block)
            binblocks[g] = InvCipher(block, longkey, 'CBC', bitiv)
            bitiv = copy.deepcopy(tempiv)
    
    if((encdec == 'dec') & (hextext == 'text')):
        output = Bin_blocks_to_Text(binblocks)
    else:
        output = Bin_blocks_to_Hex(binblocks)
    
    return(output)

## Counter Mode
def Counterm(input, hextext, encdec, key, nonce):
    bitkey = Hex_to_key(key)
    template_key = [bitarray('0'*32) for i in range(Nb*(Nr+1))]
    longkey = KeyExpansion(bitkey, template_key)
    if(encdec == 'dec'):
        binblocks = Hex_to_bin_blocks(input)
    elif(hextext == 'text'):
        binblocks = Text_to_bin_blocks(input)
    elif(hextext == 'hex'):
        binblocks = Hex_to_bin_blocks(input)
    
    for i, e in enumerate(binblocks):
        resblock = [bitarray('0'*8) for i in range(4*Nb)]
        nonceblock = Num_to_block(nonce+i+1)
        encnonce = Cipher(nonceblock, longkey, 'Counter', 'none')
        for q in range(4*Nb):
            blocksum = GF8(int.from_bytes(binblocks[i][q],"big")) \
                     + GF8(int.from_bytes(encnonce[q],"big"))
            # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
            resblock[q] = bitarray(f'{int(blocksum):08b}')
        binblocks[i] = resblock
    
    if((encdec == 'dec') & (hextext == 'text')):
        output = Bin_blocks_to_Text(binblocks)
    else:
        output = Bin_blocks_to_Hex(binblocks)
    
    return(output)




############ GRAPHICAL USER INTERFACE ############

root = Tk()
root.title('AES Implementation')

def Check_and_run(mode, text, hextext, encdec, key, iv, nonce):
    global enctext_entry
    global dectext_entry
    global valueerror
    inputsvalid = bool()
    noncevalid =  True
    ivlen = True
    try:
        testint = int(key, 16)
        if(mode == 'CBC'):
            testint = int(iv, 16)
            ivlen = (len(iv) == 8*Nb)
        elif(mode == 'Counterm'):
            nonce = int(nonce)
            noncevalid = (nonce < 2**128)
        inputsvalid = True
    except ValueError:
        inputsvalid = False

    if(len(key) == 8*Nk and ivlen and noncevalid and inputsvalid):
        valueerror.grid_forget()
        match (mode, encdec):
            case ('ECB', 'enc'):
                enctext_entry.delete(0,END)
                enctext_entry.insert(0,ECB(text, hextext, encdec, key))
            case ('ECB', 'dec'):
                dectext_entry.delete(0,END)
                dectext_entry.insert(0,ECB(text, hextext, encdec, key))
            case ('CBC', 'enc'):
                enctext_entry.delete(0,END)
                enctext_entry.insert(0,CBC(text, hextext, encdec, key, iv))
            case ('CBC', 'dec'):
                dectext_entry.delete(0,END)
                dectext_entry.insert(0,CBC(text, hextext, encdec, key, iv))
            case ('Counterm', 'enc'):
                enctext_entry.delete(0,END)
                enctext_entry.insert(0,Counterm(text, hextext, encdec, key, nonce))
            case ('Counterm', 'dec'):
                dectext_entry.delete(0,END)
                dectext_entry.insert(0,Counterm(text, hextext, encdec, key, nonce))
    else:
        valueerror.grid(column=1, row=3, sticky=(N, W, E, S), columnspan=3)


def UpdateMode(mode):
    global ivvalue_entry
    global nonce_entry
    global ainputlabel
    match mode:
        case 'ECB':
            nonce_entry.grid_forget()
            ivvalue_entry.grid_forget()
            ainputlabel.config(text='')
        case 'CBC':
            nonce_entry.grid_forget()
            ivvalue_entry.grid(column=2, row=5, sticky=(N, W, E, S), columnspan=2, pady='0 20')
            ainputlabel.config(text='IV:')
        case 'Counterm':
            ivvalue_entry.grid_forget()
            nonce_entry.grid(column=2, row=5, sticky=(N, W, E, S), columnspan=2, pady='0 20')
            ainputlabel.config(text='Nonce:')


mainframe = ttk.Frame(root, padding="8 8 8 8")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S), columnspan=3)
hextextframe = ttk.Frame(root, padding="8 8 8 8")
hextextframe.grid(column=0, row=1, sticky=(N, W, E, S))
settingsframe = ttk.Frame(root, padding="8 8 8 8")
settingsframe.grid(column=2, row=1, sticky=(N, E, S))
root.columnconfigure(2, weight=1)
root.rowconfigure(1, weight=1)

declabel = ttk.Label(mainframe, text='Klartext', justify='center')
declabel.grid(column=0, row=0, sticky=(N, W, E, S))

enclabel = ttk.Label(mainframe, text='Schlüsseltext', justify='center')
enclabel.grid(column=4, row=0, sticky=(N, W, E, S), padx='20 0')

dectext = StringVar()
dectext_entry = ttk.Entry(mainframe, textvariable=dectext, width=120)
dectext_entry.grid(column=0, row=1, sticky=(N, W, E, S), rowspan=6, padx='0 20')

enctext = StringVar()
enctext_entry = ttk.Entry(mainframe, textvariable=enctext, width=120)
enctext_entry.grid(column=4, row=1, sticky=(N, W, E, S), rowspan=6, padx='20 0')

key = StringVar()
key_entry = ttk.Entry(mainframe, textvariable=key, width=40)
key_entry.grid(column=2, row=2, sticky=(N, W, E, S), pady='40 0', columnspan=2)

valueerror = ttk.Label(mainframe, text=f'Eingabewerte sind nicht valide!', width=40, justify='center', foreground='red')

keylabel = ttk.Label(mainframe, text='Schlüssel:', justify='center')
keylabel.grid(column=1, row=2, sticky=(N, W, E, S), pady='40 0')

modeselector = StringVar()

ECBmode = Radiobutton(mainframe, text='ECB', variable=modeselector, value='ECB', command=(lambda: UpdateMode('ECB')))
ECBmode.grid(column=1, row=4, stick=(N, W, E, S))

CBCmode = Radiobutton(mainframe, text='CBC', variable=modeselector, value='CBC', command=(lambda: UpdateMode('CBC')))
CBCmode.grid(column=2, row=4, stick=(N, W, E, S))

CNTRmode = Radiobutton(mainframe, text='Counter Mode', variable=modeselector, value='Counterm', command=(lambda: UpdateMode('Counterm')))
CNTRmode.grid(column=3, row=4, stick=(N, W, E, S))

ivvalue = StringVar()
ivvalue_entry = ttk.Entry(mainframe, textvariable=ivvalue, width=40)

nonce = StringVar()
nonce_entry = ttk.Entry(mainframe, textvariable=nonce, width=40)

ainputlabel = ttk.Label(mainframe, justify='center')
ainputlabel.grid(column=1, row=5, sticky=(N, W, E, S), pady='0 20')

hextextselector = StringVar()

hexselect = Radiobutton(hextextframe, text='Hexadezimal', variable=hextextselector, value='hex', width=10)
hexselect.grid(column=0, row=0, stick=(N, W, E, S))

textselect = Radiobutton(hextextframe, text='Text', variable=hextextselector, value='text', width=7)
textselect.grid(column=1, row=0, stick=(N, W, E, S))

tutorbutton = ttk.Button(settingsframe, width=20, text='Tutor', command=(lambda: Tutor()))
tutorbutton.grid(column=2, row=0, sticky=(N, W, E, S), padx='5 0')

generatebutton = ttk.Button(settingsframe, width=20, text='Schlüssel erzeugen', command=(lambda: KeyGen()))
generatebutton.grid(column=1, row=0, sticky=(N, W, E, S), padx='5 5')

settingsbutton = ttk.Button(settingsframe, width=20, text='Einstellungen', command=(lambda: Settings()))
settingsbutton.grid(column=0, row=0, sticky=(N, W, E, S), padx='0 5')

encbutton = ttk.Button(mainframe, width=7, text='Verschlüsseln', command=(lambda: Check_and_run(modeselector.get(), dectext.get(), hextextselector.get(), 'enc', key.get(), ivvalue.get(), nonce.get())))
encbutton.grid(column=1, row=1, sticky=(N, W, E, S), columnspan=3)

decbutton = ttk.Button(mainframe, width=7, text='Entschlüsseln', command=(lambda: Check_and_run(modeselector.get(), enctext.get(), hextextselector.get(), 'dec', key.get(), ivvalue.get(), nonce.get())))
decbutton.grid(column=1, row=6, sticky=(N, W, E, S), columnspan=3)

def KeyGen():
    # Siehe https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
    randkey = [[bitarray(f'{(secrets.randbits(8)):08b}') for i in range(Nk*4)]]
    randkeyhex = Bin_blocks_to_Hex(randkey)
    randiv = [[bitarray(f'{(secrets.randbits(8)):08b}') for i in range(4*Nb)]]
    randivhex = Bin_blocks_to_Hex(randiv)
    randnonce = secrets.randbelow(2**128)
    genkeycontents = StringVar()
    genkeycontents.set(f"Neuer Schlüssel: {randkeyhex}")
    genivcontents = StringVar()
    genivcontents.set(f"Neuer Initialisierungsvektor: {randivhex}")
    gennoncecontents = StringVar()
    gennoncecontents.set(f"Neue Nonce: {randnonce}")
    newwindow = Toplevel()
    Entry(newwindow, textvariable=genkeycontents, state='readonly', readonlybackground='#f0f0f0', justify='left', relief='flat', width=80).grid(sticky=(N, W), row=0)
    Entry(newwindow, textvariable=genivcontents, state='readonly', readonlybackground='#f0f0f0', justify='left', relief='flat', width=80).grid(sticky=(N, W), row=1)
    Entry(newwindow, textvariable=gennoncecontents, state='readonly', readonlybackground='#f0f0f0', justify='left', relief='flat', width=80).grid(sticky=(N, W), row=2)

def Settings():
    global Nk
    newwindow = Toplevel()
    settingslabel = Label(newwindow, justify='left', text='Schlüssellänge verändern').grid(column=0, row=0, columnspan=3, sticky=(N, W))
    
    Nk128 = Radiobutton(newwindow, text='128', variable=Nk, value=4)
    Nk128.grid(column=0, row=1, stick=(N, W, E, S))

    Nk192 = Radiobutton(newwindow, text='192', variable=Nk, value=6)
    Nk192.grid(column=1, row=1, stick=(N, W, E, S))

    Nk256 = Radiobutton(newwindow, text='256', variable=Nk, value=8)
    Nk256.grid(column=2, row=1, stick=(N, W, E, S))

tutortexts = ['''1.: Endliche Körper
Ein Körper GF(p^m) ist eine Menge an Zahlen, wobei p eine Primzahl und m eine natürliche Zahl ist. Die Elemente sind alle Zahlen zwischen
einschließlich 0 und (p^m)-1. Sie lassen sich alle als Summen von Potenzen von p zwischen p^(m-1) und p^0 darstellen: a*p^(m-1)+...+a*p^1+a*p^0
Hier kann a alle Werte zwischen einschließlich 0 und p-1 einnehmen. Zum Beispiel enthält der endliche Körper GF(2^3) folgende Zahlen:
(0*4+0*2+0*1),(0*4+0*2+1*1),(0*4+1*2+0*1),(0*4+1*2+1*1),(1*4+0*2+0*1),(1*4+0*2+1*1),(1*4+1*2+0*1),(1*4+1*2+1*1) = 0,1,2,3,4,5,6,7

Mit einer Primzahl von 2 kann man die Koeffizienten (a) den Ziffern einer Binärzahl zuordnen: Zum Beispiel ist (1*4+0*2+1*1) = 5 = 101

In endlichen Körpern werden Addition, Subtraktion, Multiplikation und Division neu definiert: Bei der Addition und Subtraktion werden die
Koeffizienten addiert bzw. subtrahiert und mod p gerechnet. Bei der Multiplikation werden die Polynomrepräsentationen miteinander multipliziert,
die Koeffizienten mod p reduziert und das Polynom schriftlich durch ein irreduzibles Polynom dividiert. Die Division ist komplizierter; mithilfe
eines mathematischen Algorithmus wird die Zahl gefunden, die mit dem Divisor multipliziert den Wert 1 ergibt, und dann der Dividend und diese
Zahl miteinander multipliziert. All diese Berechnungen werden schnell und effizient intern vom Computer ausgetragen.''',

'''2.: AES
Der AES-Algorithmus nimmt einen Schlüssel von 128-, 192-, oder 256-Bit-Länge und einen Klartext und gibt einen Schlüsseltext heraus.
Zuerst wird der Schlüssel verlängert, dann wird der Klartext in Blöcke aufgeteilt, von denen jeder verschlüsselt wird.

Der Algorithmus hat drei Variablen: Nb, die die Blocklänge bestimmt; Nk, die die Schlüssellänge bestimmt; und Nr, die die Rundenanzahl bestimmt.
Nb ist immer gleich 4; Nk kann gleich 4, 6 oder 8 sein; und Nr ist immer gleich Nk + 6.''',

'''3.: Schlüsselverlängerung
Der Schlüssel wird auf eine Länge von 1408, 1664 oder 1920 Bits erweitert. Der erste Teil des neuen Schlüssels ist dem alten Schlüssel genau gleich.
Die restlichen Words (Ein Word = 4 Bytes = 32 Bits) werden ermittelt, in dem das Word davor mit einer Konstante Rcon in einem endlichen Körper addiert wird, die von der Wordzahl abhängig ist.
Jedes Nk-te Word wird dabei zuerst um acht Stellen bitweise rotiert und dann in die Funktion SubBytes gegeben (Siehe 5.: SubBytes).''',

'''4.: AES-Algorithmus
Für die Ver- und Entschlüsselung wird der Schlüssel in sogenannte Rundenschlüssel aufgeteilt und der Klartext wird in mehreren Runden verarbeitet.
In der ersten Runde wird der Klartextblock in einem endlichen Körper mit dem Rundenschlüssel addiert. In den restlichen werden die Funktionen
SubBytes, ShiftRows und MixColumns angewendet und dann wird der Schlüssel addiert. In der letzten Runde wird MixColumns ausgelassen.

Für die Entschlüsselung werden SubBytes und ShiftRows in der Schleife vertauscht und der Schlüssel wird vor MixColumns angewendet. Alle Funktionen
werden durch invertierte Versionen ersetzt. Wie man gut in der Abbildung sieht, wird dadurch die Reihenfolge exakt umgekehrt.''',

'''5.: SubBytes
SubBytes nimmt jedes Byte der Eingabe und multipliziert dessen Inverses in einem endlichen Körper mit der Matrix A und addiert dann mit dem Byte c.
Die inverse Version der Funktion verändert A und C.''',

'''6.: ShiftRows
ShiftRows nimmt den Block als 4*Nk-Matrix und verschiebt jede Reihe um den Reihenindex nach links. Die invertierte Funktion verschiebt nach rechts.''',

'''7.: MixColumns
Der Block wird als 4*Nk-Matrix dargestellt und jede Spalte wird in dem endlichen Körper GF(2^8) mit folgender Matrix multipliziert:
(Die umgekehrte Version hat natürlich eine andere Matrix.)''']

algorithmimg = ImageTk.PhotoImage(Image.open("AES.png"))
emptyimg = ImageTk.PhotoImage(Image.open("empty.png"))
keyimg = ImageTk.PhotoImage(Image.open("Keyschedule.png"))
cipherimg = ImageTk.PhotoImage(Image.open("Cipher.png"))
subbytesimg = ImageTk.PhotoImage(Image.open("SubBytes.png"))
shiftrowsimg = ImageTk.PhotoImage(Image.open("ShiftRows.png"))
mixcolumnsimg = ImageTk.PhotoImage(Image.open("MixColumns.png"))
tutorimages = [emptyimg,algorithmimg,keyimg,cipherimg,subbytesimg,shiftrowsimg,mixcolumnsimg]

ttindex = 0

def Tutor():
    newwindow = Toplevel()
    tutorframe = ttk.Frame(newwindow, padding="8 8 8 8")
    tutorframe.grid(column=0, row=0, sticky=(N, W, E, S))
    
    titlelabel = ttk.Label(tutorframe, text='Tutor', justify='center')
    titlelabel.grid(column=1, row=0, sticky=(N, W, E, S), padx='300 300')

    tutorlabel = ttk.Label(tutorframe, text=tutortexts[ttindex], justify='center')
    tutorlabel.grid(column=0, row=1, sticky=(N, W, E, S), columnspan=3)

    tutorimg = ttk.Label(tutorframe, justify='center', image=tutorimages[ttindex], width='3')
    tutorimg.grid(column=1, row=2, sticky=(N, W, E, S))

    lastbutton = ttk.Button(tutorframe, width=7, text='<<', command=(lambda: CycleBack(tutorlabel,tutorimg)))
    lastbutton.grid(column=0, row=0, sticky=(N, W, E, S))

    nextbutton = ttk.Button(tutorframe, width=7, text='>>', command=(lambda: CycleNext(tutorlabel,tutorimg)))
    nextbutton.grid(column=2, row=0, sticky=(N, W, E, S))

def CycleBack(label,img):
    global ttindex
    ttindex = (ttindex - 1) % 7
    label.config(text=tutortexts[ttindex])
    img.config(image=tutorimages[ttindex])

def CycleNext(label,img):
    global ttindex
    ttindex = (ttindex + 1) % 7
    label.config(text=tutortexts[ttindex])
    img.config(image=tutorimages[ttindex])

root.mainloop()

for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)
dectext_entry.focus()




############ VERWENDETE COMMANDS UND TESTS ############


### PyInstaller; im Windows-CMD-Fenster eingegeben
# pyinstaller AES.py --add-data "*.png" --windowed


### python-stopwatch; können durch Entfernen der Hashtags aktiviert werden. Vorher muss Zeile 911 auskommentiert werden.

## ECB
# ECBinput = 'The quick brown fox jumps over the lazy dog'
# ECBshortkey = '89debd08096f31abcaf0d6f6806274bd'
# ECBstopwatch = Stopwatch()
# ECBstopwatch.start()
# res = ECB(ECBinput, 'text', 'enc', ECBshortkey)
# ECBstopwatch.stop
# print(res, ECBstopwatch.elapsed)

# res = '16f2c2eb4306c016ac9c45bfb07a97d241792573e48c612b1b62ae2672ca931a8955b6147e9a186fcade1591596480be'
# ECBshortkey = '89debd08096f31abcaf0d6f6806274bd'
# ECBstopwatch = Stopwatch()
# ECBstopwatch.start()
# res = ECB(res, 'text', 'dec', ECBshortkey)
# ECBstopwatch.stop
# print(res, ECBstopwatch.elapsed)

## CBC
# CBCinput = 'The quick brown fox jumps over the lazy dog'
# CBCshortkey = '89debd08096f31abcaf0d6f6806274bd'
# CBCiv = '50e6f5dba34a1a3c752bb840b6879060'
# CBCstopwatch = Stopwatch()
# CBCstopwatch.start()
# res = CBC(CBCinput, 'text', 'enc', CBCshortkey, CBCiv)
# CBCstopwatch.stop
# print(res, CBCstopwatch.elapsed)

# res = 'a7fa324e8de4b7183bce72d882520756c033a17bd8b4debc4996ffff7294806990c66bf042705706bb695d46423f4ef3'
# CBCshortkey = '89debd08096f31abcaf0d6f6806274bd'
# CBCiv = '50e6f5dba34a1a3c752bb840b6879060'
# CBCstopwatch = Stopwatch()
# CBCstopwatch.start()
# res = CBC(res, 'text', 'dec', CBCshortkey, CBCiv)
# CBCstopwatch.stop
# print(res, CBCstopwatch.elapsed)

## Counter Mode
# CNTRinput = 'The quick brown fox jumps over the lazy dog'
# CNTRshortkey = '89debd08096f31abcaf0d6f6806274bd'
# CNTRnonce = '181736720380177676749156804959900430112'
# CNTRstopwatch = Stopwatch()
# CNTRstopwatch.start()
# res = CBC(CNTRinput, 'text', 'enc', CNTRshortkey, CNTRnonce)
# CNTRstopwatch.stop
# print(res, CNTRstopwatch.elapsed)

# res = '82b10fd820c0169f153888cb9a97afad6c442931987dc5cb79879ba95839006ee07e97151a4fcf41d3238d851b9f587e'
# CNTRshortkey = '89debd08096f31abcaf0d6f6806274bd'
# CNTRnonce = '181736720380177676749156804959900430112'
# CNTRstopwatch = Stopwatch()
# CNTRstopwatch.start()
# res = CBC(res, 'text', 'dec', CNTRshortkey, CNTRnonce)
# CNTRstopwatch.stop
# print(res, CNTRstopwatch.elapsed)