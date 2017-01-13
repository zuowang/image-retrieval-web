# -*- coding: utf-8 -*-
import cherrypy
import pickle
import urllib
import os
import numpy
#from numpy import *
from scipy import io
import cPickle


class SearchDemo:

    def __init__(self):
        # load list of images
        self.path = './street2shop/'
        self.imlist = [os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)

        input = open('feat_street2shop.pkl', 'rb')
        self.feat = cPickle.load(input)
        self.imNamelist = cPickle.load(input)
        input.close()
        self.imNum, self.dim = self.feat.shape

        # set max number of results to show
        self.maxres =20 

        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>以图搜图</title>
            <link rel="stylesheet" href="/bootstrap/css/bootstrap.min.css">
            <link rel="stylesheet" href="style.css">
            <script src='http://cdn.bootcss.com/jquery/1.11.2/jquery.min.js'></script>
            <script src='/bootstrap/js/bootstrap.min.js'></script>
            </head>
            <body>
            """
        self.footer = """
            </html>
            """

    @cherrypy.expose
    def shutdown(self):  
        cherrypy.engine.exit()

    def index(self, query=None):

        html = self.header
        html += """
                <div class="jumbotron">
                    <div class="row">
                        <div class="col-lg-4 col-lg-offset-4">
                            <div class="input-group input-group-lg">
                                <input type="text" class="form-control" placeholder="以图搜图" name="srch-term" id="srch-term">
                                    <div class="input-group-btn">
                                        <button class="btn btn-gray" type="submit"><i class="glyphicon glyphicon-search"></i></button>
                                    </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="randButton">
                    <div class="row">
                        <button class="btn btn-default center-block btn-lg btn-success" type="button"><a href='?query='>随机图片</a></button>
                        <!--<button class="btn btn-default btn-lg btn-warning" type="button"><a id="shutdown"; href="./shutdown">关闭服务器</a></button>-->
                    </div>
                </div>
            """

        if query:
            #print query
            queryID = self.imNamelist.index(query.split("/")[-1])
            queryVec = self.feat[queryID]

            score = numpy.zeros(shape=(self.imNum, 1))

            #for i,tempVec in enumerate(self.feat):
            #    score[i] = numpy.dot(queryVec, tempVec)
            #rank_ID = numpy.argsort(score[:,0])[::-1]

            score = numpy.dot(queryVec, self.feat.T)
            rank_ID = numpy.argsort(score)[::-1]
            
            # average side expansion
            queryVec = numpy.vstack((queryVec, self.feat[rank_ID[:10]])).mean(axis=0)
            score = numpy.dot(queryVec, self.feat.T)
            rank_ID = numpy.argsort(score)[::-1]

            rank_score = score[rank_ID] 

            imlist = [self.path+self.imNamelist[index] for i,index in enumerate(rank_ID[0:self.maxres])]

            for imname in imlist:
                html += "<div class='col-xs-6 col-sm-4 col-md-2 marginDown' >"
                #html += "<div class='col-sm-6 col-md-3'>"

                html += "<a href='?query="+imname+"' class='thumbnail'>"
                #html += "<a href='?query="+imname+"'>"

                html += "<img class='img-responsive' style='max-height:220px' src='"+imname+"'  />"
                #html += "<img src='"+imname+"' width='200px' height='200px' />"
                html += "</a>"
                html += "</div>"
        else:
            # show random selection if no query
            numpy.random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<div class='col-xs-6 col-sm-4 col-md-2 marginDown' >"
                #html += "<div class='col-sm-6 col-md-3'>"

                html += "<a href='?query="+imname+"' class='thumbnail'>"
                #html += "<a href='?query="+imname+"'>"

                html += "<img class='img-responsive' style='max-height:200px' src='"+imname+"'  />"
                #html += "<img src='"+imname+"' width='200px' height='200px' />"
                html += "</a>"
                html += "</div>"
                html += "</body>"
        html += self.footer
        html += """
                <footer>
                <p class='linkings'><a href='http://github.com/zuowang/'>Created by Zuo Wang</a></p>
                </footer>
                """
        return html

    index.exposed = True

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
