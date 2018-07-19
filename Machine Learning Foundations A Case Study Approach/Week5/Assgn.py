# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:39:03 2018

@author: Ananthu
"""

import graphlab
import graphlab.aggregate as agg

song_data = graphlab.SFrame('Course1/Week5/song_data.gl/')

song_data.head()

### Qn 1

df = song_data.groupby(key_columns='artist',operations={'total_count':agg.COUNT_DISTINCT('user_id')})
df[df['artist'] == 'Kanye West']
df[df['artist'] == 'Foo Fighters']
df[df['artist'] == 'Taylor Swift']
df[df['artist'] == 'Lady GaGa']

### Qn 2 & 3

popular_artist = song_data.groupby(key_columns='artist',operations={'total_count':agg.SUM('listen_count')})
popular_artist.sort('total_count', ascending = False)
popular_artist.sort('total_count')
