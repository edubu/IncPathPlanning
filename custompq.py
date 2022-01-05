# A simple implementation of Priority Queue
# using Queue.
import heapq

class CustomPQ(object):
	def __init__(self):
		self.queue = []
		heapq.heapify(self.queue)

	def __str__(self):
		return ' '.join([str(i) for i in self.queue])

	# for checking if the queue is empty
	def isEmpty(self):
		return len(self.queue) == 0

	# for inserting an element in the queue
	def put(self, data):
		heapq.heappush(self.queue, data)

	# for popping an element based on Priority
	def pop(self):
		heapq.heappop(self.queue)
	
	def get(self):
		return heapq.heappop(self.queue)

	def qsize(self):
		return len(self.queue)

