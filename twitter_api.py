from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import nlp_project as s

# Consumer key, consumer secret, access token, access secret
ckey    = "x"
csecret = "x"
atoken  = "x"
asecret = "x"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment_mod(tweet)
        print((tweet, sentiment_value, confidence))

        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["x"])
