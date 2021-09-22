from my_ens.load import load_data

def test_func():        
    x,y,clz=load_data('glass')
    print(x.shape)

if __name__ == '__main__':
    test_func()