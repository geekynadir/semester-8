
n=int(input("Enter the value of N "))
pre,next=0,1
print(f'{pre}')
print(f'{next}')
for i in range (n):
    sum=pre+next
    print(sum)
    pre=next
    next=sum
    
