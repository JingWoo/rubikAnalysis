FROM python:latest

MAINTAINER wujing <jing.woo@outlook.com>

ENV APP /root/rubikAnalysis

RUN mkdir $APP
WORKDIR $APP

RUN mkdir -p ${HOME}/.pip && touch  ${HOME}/.pip/pip.conf

RUN echo "[global]" >> ${HOME}/.pip/pip.conf && \
    echo "timeout = 6000" >> ${HOME}/.pip/pip.conf && \
    echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> ${HOME}/.pip/pip.conf && \
    echo "trusted-host = mirrors.aliyun.com" >> ${HOME}/.pip/pip.conf

COPY requirements.txt .
COPY rubikanalysis/analysis.py .

RUN pip install -r requirements.txt
RUN pip install pyinstaller

RUN pyinstaller analysis.py --onefile

CMD ["./dist/analysis"]





