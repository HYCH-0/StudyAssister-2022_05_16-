Seat_list = list(range(1,1+1))   #좌석수 1개로 가정__실제로는 더 많으나 일단 1인칭시점에서 적용된다고 가정함
student_Dict={}                   #학생 리스트
while Seat_list:
    n1=int(input('좌석번호를 입력하세요: '))
    if n1>1000 or n1<0:
        print('이용가능한 좌석이 아닙니다. 다시 입력해주세요.')
        continue
    n2 = input('이름을 입력하세요: ')
    n3 = int(input('이용하실 시간을 입력하세요(단위: 초): '))
    student_Dict[n2]=n3
    break
#수정 필요: 