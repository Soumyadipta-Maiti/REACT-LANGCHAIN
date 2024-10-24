FROM python:3.12.3
COPY . .
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT [ "python" ]
CMD [ "Functions_Langchain/testing_fastapi.py"]
