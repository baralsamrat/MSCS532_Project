
'''
    output: 
    python3 main.py 
    Matched recipient recipient2 with donor donor2.
    Matched recipient recipient3 with donor donor1.
    Matched recipient recipient1 with donor donor3.
'''

import heapq
from collections import defaultdict

# Recipient class representing the recipient details
class Recipient:
    def __init__(self, id, blood_type, urgency):
        self.id = id
        self.blood_type = blood_type
        self.urgency = urgency

    def __lt__(self, other):
        return self.urgency > other.urgency  # Higher urgency gets higher priority

# Priority queue for managing recipient urgency
class UrgencyQueue:
    def __init__(self):
        self.heap = []

    def add_recipient(self, recipient):
        heapq.heappush(self.heap, recipient)

    def get_highest_priority(self):
        return heapq.heappop(self.heap) if self.heap else None

# Donor class representing the donor details
class Donor:
    def __init__(self, id, blood_type, hla_type):
        self.id = id
        self.blood_type = blood_type
        self.hla_type = hla_type

# Hash table to manage donors by blood type
class DonorHashMap:
    def __init__(self):
        self.hash_table = defaultdict(list)

    def add_donor(self, donor):
        self.hash_table[donor.blood_type].append(donor)

    def get_donors_by_blood_type(self, blood_type):
        return self.hash_table.get(blood_type, [])

# Simple matching algorithm based on blood type and urgency
def match_donors_to_recipients(donor_map, urgency_queue):
    matches = []
    while urgency_queue.heap:
        recipient = urgency_queue.get_highest_priority()
        donors = donor_map.get_donors_by_blood_type(recipient.blood_type)
        
        if donors:
            matched_donor = donors.pop(0)  # Get the first available donor
            matches.append((recipient.id, matched_donor.id))
        else:
            print(f"No suitable donor found for recipient {recipient.id} with blood type {recipient.blood_type}.")
    
    return matches

# Test script demonstrating functionality
if __name__ == "__main__":
    # Initialize data structures
    donor_map = DonorHashMap()
    urgency_queue = UrgencyQueue()

    # Adding donors
    donor1 = Donor(id="donor1", blood_type="A", hla_type="HLA-A1")
    donor2 = Donor(id="donor2", blood_type="B", hla_type="HLA-B1")
    donor3 = Donor(id="donor3", blood_type="A", hla_type="HLA-A2")
    donor_map.add_donor(donor1)
    donor_map.add_donor(donor2)
    donor_map.add_donor(donor3)

    # Adding recipients with varying urgency
    recipient1 = Recipient(id="recipient1", blood_type="A", urgency=5)
    recipient2 = Recipient(id="recipient2", blood_type="B", urgency=10)
    recipient3 = Recipient(id="recipient3", blood_type="A", urgency=8)
    urgency_queue.add_recipient(recipient1)
    urgency_queue.add_recipient(recipient2)
    urgency_queue.add_recipient(recipient3)

    # Matching donors to recipients
    matches = match_donors_to_recipients(donor_map, urgency_queue)
    for recipient_id, donor_id in matches:
        print(f"Matched recipient {recipient_id} with donor {donor_id}.")
