import bz2

    
def memoryNCD(data1, data2, compressionLevel = 9):

    C_o1 = len(bz2.compress(data1, compresslevel = compressionLevel))
    C_o2 = len(bz2.compress(data2, compresslevel = compressionLevel))
    C_o1o2 = len(bz2.compress(data1 + data2, compresslevel = compressionLevel))
    
    f_ncd = float(C_o1o2 - min(C_o1, C_o2)) / max(C_o1, C_o2)
    
    dictFileNCD = {}
    dictFileNCD['Compression Level'] = compressionLevel
    dictFileNCD['C_o1'] = C_o1
    dictFileNCD['C_o2'] = C_o2
    dictFileNCD['C_o1o2'] = C_o1o2
    dictFileNCD['NCD'] = f_ncd
    
    return dictFileNCD


def fileNCD(inputFn1, inputFn2, compressionLevel = 9):

    f1 = open(inputFn1, 'rb')
    data1 = f1.read()
    f1.close()
    
    f2 = open(inputFn2, 'rb')
    data2 = f2.read()
    f2.close()
    
    return memoryNCD(data1, data2, compressionLevel)

