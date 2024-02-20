# import pymongo

# # Check the TLS/SSL version supported by pymongo
# tls_version = pymongo.ssl.OPENSSL_VERSION
# print("TLS/SSL version supported by pymongo:", tls_version)


from pymongo import MongoClient

# Connect to MongoDB (replace the connection string with your own)
client = MongoClient("mongodb+srv://amanagrawal:ofSqQM7JfhkQfPPi@fushionnet0.crmusbk.mongodb.net/?retryWrites=true&w=majority")

# Access the underlying SSL/TLS context
ssl_context = client._get_topology().select_server_by_address(("localhost", 3001))._socket_for_writes._ssl_context

# Get the protocol version
tls_version = ssl_context.get_protocol_name()
print("TLS/SSL version used by pymongo:", tls_version)
