import json
import os.path
import random
import urllib.request as urllib2
from time import sleep


def create_post_url(graph_url, token):
    # create authenticated post URL
    post_args = "/posts/?key=value&access_token=" + token
    post_url = graph_url + post_args
    return post_url


def render_to_json(graph_url):
    # render graph url call to JSON
    web_response = urllib2.urlopen(graph_url)
    readable_page = web_response.read()
    json_data = json.loads(readable_page.decode("utf-8"))
    return json_data


def main():
    # Tokens expire - get new one here: https://developers.facebook.com/tools/explorer
    APP_TOKEN = "EAACEdEose0cBAA88rhfrBFjui2jZCOKouY0cDWIxovb7IbK9U2ZCLBvwR7UqqDzKnr6nJsCoCMWJL8nZBmJIN9j00dSv4StPkgj2EzaZC50L15TjezF6VCVNKE75cl03OluG4ilsW1ueeIgPJ7O8g0ZBZCUZCZACV5ZCP4xP7UcudgSwMRKB3W1pcgFlwNW9krrQZD"
    USER = "YairLapid"
    INPUT_FOLDER = "input"

    graph_url = "https://graph.facebook.com/"

    current_page = graph_url + USER
    post_url = create_post_url(current_page, APP_TOKEN)

    char_counter = counter = 0

    while post_url is not None and char_counter < 15000000:
        # extract post data
        json_postdata = render_to_json(post_url)
        json_fbposts = json_postdata['data']

        # print post messages and ids
        for post in json_fbposts:
            if 'message' in post:
                length = len(post["message"])
                fileName = os.path.join(os.curdir, INPUT_FOLDER, post["id"] + '.txt')
                fd = open(fileName, 'w', encoding="utf-8")
                # TODO: write tries to encode to ascii and fails - need to convert from Hebrew encoding to range(128).
                fd.write(post["message"])
                # cPickle.dump(post['message'], fd)
                print("msg #%d: length %d, total length %d" % (counter, length, char_counter))
                counter += 1
                char_counter += length

        post_url = json_postdata['paging'].get('next') if 'paging' in json_postdata else None
        sleep_time = random.uniform(0.5, 1.0)
        # sleep(sleep_time)


if __name__ == "__main__":
    main()
