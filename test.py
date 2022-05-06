
# If you need to import additional packages or classes, please import here.

def func():
    line = input().split()
    m = int(line[0])
    n = int(line[1])
    x = int(line[2])
    print(m,n,x)
    mianshiguan =[]
    mianshizhe = []
    for i in range(m):
        mianshiguan[i]= input().split()
    for i in range(n):
        mianshizhe[i] = input().split()    

    print(mianshiguan)
    print(mianshizhe)
    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().

if __name__ == "__main__":
    func()
