FROM public.ecr.aws/lambda/python:3.9

WORKDIR /var/task

COPY lambda.py functions.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt --target .

CMD ["lambda.lambda_handler"]