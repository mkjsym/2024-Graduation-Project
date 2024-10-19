import smtplib
from email.mime.text import MIMEText

#use port 587 or 465
smtp = smtplib.SMTP('smtp.gmail.com', 587)

smtp.ehlo()
smtp.starttls()

#sender email, sender's app password
smtp.login('mkjsym@gmail.com', 'itof hbrd duzh vwjw')

msg = MIMEText('content: test')
msg['Subject'] = 'title: test msg'

smtp.sendmail('mkjsym@gmail.com', 'mkjsym@naver.com', msg.as_string())

smtp.quit()
