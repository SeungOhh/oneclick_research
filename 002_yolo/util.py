from twilio.rest import Client

# pip install twilio


#%%
# https://console.twilio.com/us1/develop/phone-numbers/manage/incoming

#%%
def send_sms(phone_number_to, messages):
    phone_number_from = "+12185597312"
    
    # Find your Account SID and Auth Token at twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = ""
    auth_token = ""
    client = Client(account_sid, auth_token)
    
    message = client.messages \
                    .create(
                         body = messages,
                         from_= phone_number_from,
                         to=phone_number_to
                     )
    
    print(message.sid)



#%%
if __name__ == "__main__":
    phone_number_to = "+821035278108"
    messages = "[동양미래대학교 팀] 현재 위성 데이터 분석결과 인구밀도가 높은 '위험' 지역을 발견했습니다. 조속히 대비해주시기 바랍니다. "
    send_sms(phone_number_to, messages)


