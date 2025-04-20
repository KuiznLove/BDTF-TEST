import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr

class QQMailSender:
    # def __init__(self, sender_email, sender_password, receiver_email):
    def __init__(self):
        self.sender_email = '1665813376@qq.com'
        self.sender_password = 'frxlafepjugafage'
        self.receiver_email = '1665813376@qq.com'
        self.smtp_server = 'smtp.qq.com'
        self.smtp_port = 465

    def send_mail(self, data_dir, seed, pair_f1, pair_p, pair_r):
        if data_dir == '../data/ASTE/V2/14lap':
            f1, p, r = 0.6174, 0.6894, 0.5597
        elif data_dir == '../data/ASTE/V2/14res':
            f1, p, r = 0.7435, 0.7553, 0.7324
        elif data_dir == '../data/ASTE/V2/15res':
            f1, p, r = 0.6612, 0.6876, 0.6371
        elif data_dir == '../data/ASTE/V2/16res':
            f1, p, r = 0.7227, 0.7144, 0.7313
        else:
            f1, p, r = 0.0, 0.0, 0.0

        subject = f'data={data_dir} (seed={seed}) 已训练完成'
        content = (f'pair_p  = {pair_p*100:.2f}\t delta = {(pair_p-p)*100:.2f},\n'
                   f'pair_r  = {pair_r*100:.2f}\t delta = {(pair_r-r)*100:.2f},\n'
                   f'pair_f1 = {pair_f1*100:.2f}\t delta = {(pair_f1-f1)*100:.2f},\n')

        # 构造邮件
        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = formataddr(('发件人', self.sender_email))  # 使用 formataddr 规范 From 字段
        message['To'] = formataddr(('收件人', self.receiver_email))  # 使用 formataddr 规范 To 字段
        message['Subject'] = Header(subject, 'utf-8')

        try:
            # 连接QQ邮箱的SMTP服务器
            smtp_obj = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            smtp_obj.login(self.sender_email, self.sender_password)
            smtp_obj.sendmail(self.sender_email, self.receiver_email, message.as_string())
            smtp_obj.quit()
            print("邮件发送成功")
        except smtplib.SMTPException as e:
            print(f"邮件发送失败: {e}")

# 使用示例
if __name__ == "__main__":
    sender_email = '1665813376@qq.com'
    sender_password = 'frxlafepjugafage'  # 注意：这里需要填写QQ邮箱的授权码，而不是密码
    receiver_email = '1665813376@qq.com'

    mail_sender = QQMailSender()
    mail_sender.send_mail(data_dir='14lap', seed=12345, pair_f1=0.5746887966804979, pair_p=0.6548463356973995, pair_r=0.512014787430684)