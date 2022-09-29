import logging
import sys
import time
import boto3
import os

from src.job_queue.message_wrapper import receive_messages, delete_message
from src.job_queue.queue_wrapper import get_queue, get_queues, remove_queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobQueue:
    """Keep track of the task that the parser is working on to skip it within the retry job container."""

    def __init__(self):
        self.sqs = boto3.resource("sqs", region_name="us-east-1")
        self.sqs_client = boto3.client("sqs", region_name="us-east-1")

        run_dir = os.environ.get("run_dir", "runs/")
        aln_run_dir = "".join(c for c in run_dir if c.isalnum())
        self.queue_name = aln_run_dir + "_parser_job_queue"

        queues = get_queues()
        queues = [i.split("/")[-1] for i in queues]

        if self.queue_name not in queues:
            logger.info("Creating queue '%s'", self.queue_name)
            self.queue = self.sqs.create_queue(QueueName=self.queue_name)

        self.queue = get_queue(self.queue_name)

        logger.info("Got queue '%s' with URL=%s", self.queue_name, self.queue.url)

    def skip(self, task_id: str) -> bool:
        """
        Check whether the task is in the queue for the parser.

        If it is we assume it should be skipped as it would have caused the parser to fail.
        """
        current_tasks = self.read_messages_()
        for task in current_tasks:
            if task_id == task.body:
                logger.info("Task %s is in the queue, skipping and deleting.", task_id)
                self.delete(task_id)
                return True
        self.send_message_(task_id)
        logger.info("Task %s is not in the queue, therefore and processing.", task_id)
        return False

    def delete(self, task_id):
        """Delete a message from the queue."""
        logger.info("Deleting task %s from the queue.", task_id)
        messages = self.read_messages_()
        messages_to_delete = []

        for message in messages:
            if message.body == str(task_id):
                messages_to_delete.append(message)

        if messages_to_delete:
            for i in messages_to_delete:
                response = delete_message(i)
                print(response)

    def send_message_(self, task_id):
        """Send a message to the queue."""
        logger.info("Sending task %s to the queue.", task_id)
        response = self.sqs_client.send_message(
            QueueUrl=self.queue.url, DelaySeconds=10, MessageBody=(str(task_id))
        )

        return response

    def read_(self):
        """Read messages from the queue whilst handling timeouts from queue."""
        messages = []

        batch_size = 10
        more_messages = True
        while more_messages:
            received_messages = receive_messages(self.queue, batch_size, 2)
            sys.stdout.flush()
            for message in received_messages:
                messages.append(message)
            if received_messages:
                pass
            else:
                more_messages = False

        return messages

    def read_messages_(self):
        """Read messages from the queue."""
        logger.info("Reading messages from the queue.")
        messages = self.read_()
        counter = 1
        while not messages:
            time.sleep(counter * 10)
            messages = self.read_()
            counter += 1
            if counter > 3:
                break

        return messages

    def delete_queue(self):
        """Delete the queue."""
        logger.info("Deleting queue '%s'", self.queue_name)
        remove_queue(self.queue)
