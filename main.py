import glob
import re
import random
import cv2
from PIL import Image
from skimage.io import imread
import numpy as np
import operator
import matplotlib.pyplot as plt
import requests, urllib
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

import os; os.chdir("C:/Users/mike/Desktop/hack")

def _load_emoticons(path = "graphics"):
    files = glob.glob(path + "/*.png")
    emotions = [re.search('\\\\(.+?)[0-9].png', x).group(1) for x in files]
    return emotions, [Image.fromarray(imread(file)) for file in files]

emots = _load_emoticons()

def pick_emoticons(emotion, emots):
    return random.choice([emots[1][i] for i, x in enumerate(emots[0]) if x == emotion])

def process_image(url = None, urldata = None, key = '6e683b8651bf495292f2eda0e69ac4b2', json = True, params = urllib.urlencode({}), render = True, show = False, **kwargs):
    if url is not None and json:
        headers = {'Ocp-Apim-Subscription-Key': key,
                   'Content-Type': 'application/json'}        
        response = requests.request('post', 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize',
                                    json = {"url": url},
                                    data = None,
                                    headers = headers,
                                    params = params)
    else:
        if urldata is None:
            urldata = open(url, "rb").read()            
        headers = {'Ocp-Apim-Subscription-Key': key,
                   'Content-Type': 'application/octet-stream'}
        response = requests.request('post', 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize',
                                    json = None,
                                    data = urldata,
                                    headers = headers,
                                    params = params)
    
    if response.status_code == 200 or response.status_code == 201:
        if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
            result = None 
        elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
            if 'application/json' in response.headers['content-type'].lower(): 
                result = response.json() if response.content else None 
            elif 'image' in response.headers['content-type'].lower(): 
                result = response.content
    else:
        print("Error code: %d" % (response.status_code))
        print("Message: %s" % ( response.json()['error']['message']))
    if render:
        img = render_image(url = url, urldata = urldata, annotation = result, json = json, show = show, **kwargs)
        return img
    else:
        return result

def render_image(img = None, url = None, urldata = None, annotation = None, json = True, show = True, alpha = (0, 1)):
    if img is None:
        if url is not None and json:
            arr = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
        elif urldata is not None:
            arr = np.asarray(bytearray(urldata), dtype = np.uint8)
        else:
            urldata = open(url, "rb").read()
        img = cv2.cvtColor(cv2.imdecode(arr, -1), cv2.COLOR_BGR2RGB)
    if type(img).__name__ != "Image":
        img = Image.fromarray(img)
    for currFace in annotation:
        faceRectangle = currFace['faceRectangle']
        currEmotion = max(currFace['scores'].items(), key=operator.itemgetter(1))[0]
        facescale = (1 + float(faceRectangle['width'] * faceRectangle['height']) / float(img.size[0] * img.size[1]))**2
        w = int(faceRectangle['width'] / facescale)
        x, y = (faceRectangle['left'] - w // 2, int(faceRectangle['top'] - 2 * w))
        if y < -w:
            x += np.sign(img.size[0] / 2 - x) * w
            y += w
        _emot = np.array(pick_emoticons(currEmotion, emots).resize((w, w)))
        _emot[:,:,-1] = _emot[:,:,-1] - np.random.randint(alpha[0], alpha[1], size = (w, w))
        _emot = Image.fromarray(_emot)
        img.paste(_emot, (x + w // 2, y + w // 2), _emot)
    if show:
        ig, ax = plt.subplots(figsize=(15, 20))
        ax.imshow(img)
    return img

def render_video(url, outfile = "tmp.mp4", fps = 15, prevlayer = None, passes = 1, fadeout = None, fadein = None, step = None):
    vid = VideoFileClip(url)
    layers = [vid]
    if prevlayer is not None:
        layers = layers + prevlayer
    #tminmax = (int(min(4, vid.duration / fadetime - 2)), int(vid.duration / fadetime))
    for p in range(passes):
        #ov = []
        if step is None:
            tslices = np.linspace(np.random.uniform(0, 2), vid.duration - np.random.uniform(0, 2), num = np.random.randint(4, 20))
        else:
            tslices = np.linspace(np.random.uniform(0, 2), vid.duration - np.random.uniform(0, 2), num = int(vid.duration / step))
        #tslices = np.linspace(1.5, vid.duration, num = 2)
        for i, t in enumerate(tslices):
            print(p, i, len(tslices))
            blank = Image.new("RGBA", (vid.size[0], vid.size[1]))
            img = cv2.cvtColor(vid.get_frame(t), cv2.COLOR_BGR2RGB)
            urldata = cv2.imencode(".png", img)[1].tobytes()
            ann = process_image(urldata = urldata, render = False, json = False)
            clipimg = np.asarray(render_image(img = blank, annotation = ann, show = False))
            #clip = ImageClip(clipimg, duration = t - tslices[i - 1], transparent = True)
            clip = ImageClip(clipimg, duration = max(t - tslices[i - 1], fadeout, fadein)).set_start(t)
            if fadein:
                clip = clip.crossfadein(fadein)
            if fadeout:
                clip = clip.crossfadeout(fadeout)
            layers.append(clip)
        #ov = concatenate_videoclips(ov)
        #layers.append(ov)
    vidout = CompositeVideoClip(layers).set_fps(fps)
    vidout.write_videofile(outfile)
    return vidout, layers[1:]

a = process_image('http://sims.ess.ucla.edu/people-images/2016_Mar_group.jpg', show = True)
a = process_image("targets/testtarget.jpg", json = False, show = True)
a = process_image("targets/target2.png", json = False, show = True)
a = process_image("targets/target3.jpg", json = False, show = True)

vidname = ["anger_hitler_1", "anger_hitler_2", "happiness_mm", "sadness_rocky", "sadness_tobey", "sadness_tomcruise", "surprise_happiness_kid"]
vidname = ["anger_hitler_1"]
for vidname in vidname:
     vv, ll = render_video("targets/" + vidname + ".mp4", outfile = "targets/" + vidname + "tagged.mp4", passes = 1, step = 0.1, fadeout = 1)#, prevlayer = ll)
