import collections
import json
import locust
import random
import time

FILENAME_LABELS = [
    ('http://67.media.tumblr.com/387b75e9d5c1181c3ac6bac26a976010/tumblr_o9dq1veOz61vygv0ao1_1280.jpg', 'beer'),
    ('http://66.media.tumblr.com/823a6c1c377eb424885b68353167ef31/tumblr_nwgzihD06F1rzngl6o2_500.jpg', 'bisque'),
    ('http://67.media.tumblr.com/tumblr_lhtq656uMF1qd45ayo1_500.jpg', 'bread'),
    ('http://67.media.tumblr.com/tumblr_m9j1rizwcT1qdei8m.jpg', 'broth'),
    ('http://66.media.tumblr.com/be962e038ae8a2497b955e4a1e656d78/tumblr_inline_n6cdgoQF6p1sf3bgt.jpg', 'bun'),
    ('http://65.media.tumblr.com/f8ec1030619dfb47e3eae41b7260ffbd/tumblr_inline_ngzoaxILF91suhfvf.jpg', 'cakeandcookie'),
    ('http://66.media.tumblr.com/b756fd3bebf87d820da0854acbf3a0f4/tumblr_inline_n0qghfjmdo1ri8bh1.jpg', 'canned_drink'),
    ('http://66.media.tumblr.com/80545ea2afdea87e3de5dc7db6cc69eb/tumblr_mzke7oBzyE1qa8q3yo1_500.jpg', 'cocktail'),
    ('http://65.media.tumblr.com/f49706468c7a7a71da73765391dbbfc8/tumblr_inline_n2y0qkpHxd1sap76t.jpg', 'coffee'),
    ('http://65.media.tumblr.com/tumblr_m5mluqpxui1qjlt3to1_500.jpg', 'curry'),
    ('http://67.media.tumblr.com/5c08267e459f428e4b750d77ab5c2df7/tumblr_n4l7gu2OHE1txqnilo1_1280.jpg', 'dessert'),
    ('http://67.media.tumblr.com/00e642948953d511fa09cb64b012008b/tumblr_inline_o7al5nKMKw1u4bf7h_1280.jpg', 'dim_sum'),
    ('http://66.media.tumblr.com/3f357270cabb31e717e12204640b7cb5/tumblr_n9syxcyJpD1qczjobo1_1280.jpg', 'dumpling'),
    ('http://66.media.tumblr.com/tumblr_l968y5nCWq1qcf27qo1_1280.jpg', 'egg'),
    ('http://65.media.tumblr.com/36a397fdcbc05ec99549a8e8747d895e/tumblr_n5mlrf5V4x1qaqbato2_r1_1280.jpg', 'fried'),
    ('http://65.media.tumblr.com/d87d2389b722ef7eaa8e84a7f93b3186/tumblr_inline_mr375elrSM1qz4rgp.jpg', 'hamburger'),
    ('http://67.media.tumblr.com/a3ee2e5e0d3c28ab4ab73514d6a5848b/tumblr_inline_odds2gfNTl1t957cg_1280.jpg', 'hand_drink'),
    ('http://67.media.tumblr.com/41cc3019f8ac3f3674a06d12cff47bc0/tumblr_inline_nrsn5zadwK1slqxr7_500.jpg', 'ice_cream'),
    ('http://65.media.tumblr.com/bb236703e4199b2e2585c358c8dce887/tumblr_n4teklOxxr1sgm4feo1_500.jpg', 'meat'),
    ('http://67.media.tumblr.com/60548a744d01987c2f3850075b1610a5/tumblr_nu0mm8mgn31r29uexo1_500.jpg', 'noodle'),
    ('http://66.media.tumblr.com/1b4e9e33f7a90e85eb6dd7b490559596/tumblr_mtbxrs34nv1qczxz7o1_1280.jpg', 'pancakes'),
    ('http://66.media.tumblr.com/d9fce2077705258e18fa128ec8c165c7/tumblr_ns1cgvdslj1ubks3jo1_1280.jpg', 'pizza'),
    ('http://66.media.tumblr.com/91a6d98f41230f28a3a51f1e22b5e649/tumblr_n4sv4v8Prn1svn89to2_1280.jpg', 'pot'),
    ('http://67.media.tumblr.com/530946ac1f90087098000175a5f41bc6/tumblr_o7bsr89MhS1r0q9woo1_1280.jpg', 'pure_wine'),
    ('http://67.media.tumblr.com/ea5c1d1b704d041a4dcdb627db0bedda/tumblr_nq1mundodG1s04h2ho1_1280.jpg', 'rice'),
    ('http://67.media.tumblr.com/0aed74136529baf9efa5d3dc0bf6e4a1/tumblr_inline_na00qs1NyH1s17clb.jpg', 'sake'),
    ('http://66.media.tumblr.com/31775972691398a2b44a067dc83352b0/tumblr_n4e5ia3DnM1rodxmao1_500.jpg', 'salad'),
    ('http://66.media.tumblr.com/40bc261aa18baa0c351fc3fad90722aa/tumblr_nb6uubPRSv1rwy6s8o1_500.jpg', 'seafood'),
    ('http://67.media.tumblr.com/c7fb60dd0f721296274b913d61d42c22/tumblr_inline_o2sk3anodd1tyob3a_500.jpg', 'snacking'),
    ('http://67.media.tumblr.com/60cc11f4cb2c86c703164545cd67ad59/tumblr_n22v3s5Jra1qkyzm3o1_1280.jpg', 'street_food'),
    ('http://67.media.tumblr.com/32c708c1960596dc6d71fc91fe68dc41/tumblr_o9hxevUb9v1vo76weo1_500.jpg', 'sushi'),
    ('http://67.media.tumblr.com/f3decdae7b7b1cadfd1ff4e0084926f0/tumblr_mj211ioeSI1rnr968o1_500.jpg', 'tea'),
    ('http://65.media.tumblr.com/6606b8346b659f775a88649616b87257/tumblr_objwzunzzj1rfubseo1_500.jpg', 'vegetables'),
    ('http://67.media.tumblr.com/57d05e3d0aa2b6630e83a337c31fc013/tumblr_ob4ecifAHE1vovmnvo1_500.jpg', 'vodka'),
    ('http://66.media.tumblr.com/36baaf044a45f76c2eda75d7ced7faf8/tumblr_mzgh3cZIwL1sko5g4o1_1280.jpg', 'whiskey'),
    ('http://65.media.tumblr.com/08f50b3b24fa7938bfbfa237625b0518/tumblr_mw8aewhCf51rkgoufo1_500.jpg', 'wine'),
    ('http://67.media.tumblr.com/06e5b82913e1f37690fb27af7cf25c8e/tumblr_inline_oa3k42QmDl1tylhqi_540.jpg', 'yakitori'),
]


class TaskSet(locust.TaskSet):
    @locust.task
    def post(self):
        num_filenames = random.randint(20, 30)
        filename_labels = random.sample(FILENAME_LABELS, num_filenames)
        (file_names, labels) = zip(*filename_labels)

        start = time.time()
        response = self.client.post('/classify/food_type', json={'images': file_names})
        print('{:.4f} sec'.format(time.time() - start))
        items = json.loads(response.text, object_hook=collections.OrderedDict)
        for (item, file_name, label) in zip(items, file_names, labels):
            (class_name, prob) = sorted(item['classes'].items(), key=lambda x: x[1])[-1]
            if label != class_name:
                print('{} -> {}({})'.format(label, class_name, float(prob)))


class User(locust.HttpLocust):
    task_set = TaskSet
    min_wait = 5000
    max_wait = 20000
