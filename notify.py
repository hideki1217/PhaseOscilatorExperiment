from pathlib import Path
import sys
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from pathlib import Path
import yaml

def _send_mail(gmail, password, mail, mailText, subject):
    charset = 'iso-2022-jp'
    msg = MIMEText(mailText, 'plain', charset)
    msg['Subject'] = Header(subject.encode(charset), charset)
    smtp_obj = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_obj.ehlo()
    smtp_obj.starttls()
    smtp_obj.login(gmail, password)
    smtp_obj.sendmail(gmail, mail, msg.as_string())
    smtp_obj.quit()

def email_me(subject, body):
    cwd = Path(__file__).absolute().parent
    with open(cwd / "config.yaml") as f:
        config = yaml.safe_load(f)

    gmail = config["gmail"]
    password = config["gmail_pass"]
    mail = config["gmail"]
    _send_mail(gmail, password, mail, body, subject)

if __name__ == "__main__":
    assert len(sys.argv) == 2
    path = Path(sys.argv[1])

    with open(path / "result.log") as f:
        content = '\n'.join(f.readlines())
        email_me(
        '[phase] Notification of end of execution',
        f"------- result.log -------\n{content}\n------------------")