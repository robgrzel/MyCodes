
m = 28


if m==5:
    n = 4

    if n==0:
        from package.module1 import *
        print (vars())
        print (globals())

        print(module1, module2, A,B,C)

    if n==1 or n==2 or n==3:
        from package import *
        print (vars())
        print (globals())

        print(module1, module2, module3)
        print(A,B,C,D,E,F,G)

    if n==4:
        from package.module1 import *

        print(vars())
        print(globals())
        print(A,B,E)

from importlib import reload # Python 3.4+ only
if m == 28:
    import foo
    foo.var = 9

    n = 4

    if n == 0:
        pass#from "foo" reload(var)
    if n== 1:
        import foo
        print(foo.var)

    if n==2:
        pass#reimport foo

    if n==3:
        import foo.var
        print(foo.var)

    if n==4:
        reload(foo)
        print(foo.var)