seat_list=list(range(1,100+1))
student_dict={}
def admin():
    n1 = int(input('좌석번호를 입력하세요: '))
    if n1 not in seat_list:
        print('이용가능한 좌석이 아닙니다. 다시 입력해주세요.')
    n2 = input('이름을 입력하세요: ')
    n3 = int(input('이용하실 시간을 입력하세요: '))
    student_dict[n2] = n3
    seat_list.remove(n1)
while seat_list:
    admin()