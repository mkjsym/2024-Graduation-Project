import smtplib
from email.mime.text import MIMEText

#use port 587 or 465
smtp = smtplib.SMTP('smtp.gmail.com', 587)

smtp.ehlo()
smtp.starttls()

#sender email, sender's app password
smtp.login('mkjsym@gmail.com', 'itof hbrd duzh vwjw')

msg = MIMEText('안녕하세요 서경대학교 컴퓨터공학과 2019305065학번 전영민입니다.')
msg['Subject'] = '졸업작품 2 smtp를 사용한 이메일 전송 기능 구현'

smtp.sendmail('mkjsym@gmail.com', 'mkjsym@naver.com', msg.as_string())

smtp.quit()
