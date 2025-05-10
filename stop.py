from langsmith import Client

client = Client()  # reads LANGSMITH_API_KEY
client.cancel_run(run_id="a7058d11-6bcb-46ec-a7cf-fd21eae40ac7")
print("Cancellation requested.")