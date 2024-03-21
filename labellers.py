from seizure_data import SeizureData


class Labeller:
    def __init__(self):
        ...

    def label(self, seizure_data: SeizureData, timestamp):
        ...

    def reset(self):
        ...


class Pre_Ictal_Labeller(Labeller):
    def __init__(self, seconds_before: int, balanced: bool):
        self.seconds_before = seconds_before
        self.curr_seizure = 1
        self.negative_available = -1
        self.balanced = balanced

    def label(self, seizure_data: SeizureData, timestamp):
        if self.negative_available == -1:
            self.negative_available = self.seconds_before * seizure_data.seizure_count
        # if not in the x seconds before a seizure to the end of the seizure, label it 0 (not pre-ictal)
        if not seizure_data.start_end[self.curr_seizure][0] - self.seconds_before < timestamp < seizure_data.start_end[self.curr_seizure][1] \
            and (self.negative_available > 0 or not self.balanced):
            self.negative_available -= 1
            return 0
        # else if in the x seconds before a seizure, label it 1 (pre-ictal)
        elif seizure_data.start_end[self.curr_seizure][0] - self.seconds_before <= timestamp < seizure_data.start_end[self.curr_seizure][0]:
            return 1
        
        if timestamp > seizure_data.start_end[self.curr_seizure][1] - self.seconds_before and self.curr_seizure < len(seizure_data.start_end):
            self.curr_seizure += 1


    def reset(self):
        self.curr_seizure = 1
        self.negative_available = -1


class IctalLabeller(Labeller):
    def __init__(self, balanced: bool):
        self.curr_seizure = 1
        self.negative_available = -1
        self.balanced = balanced
    
    def label(self, seizure_data: SeizureData, timestamp):
        if self.negative_available == -1 and self.balanced:
            for seizure in seizure_data.start_end.values():
                self.negative_available += seizure[1] - seizure[0]

        if not seizure_data.start_end[self.curr_seizure][0] < timestamp < seizure_data.start_end[self.curr_seizure][1] \
            and (self.negative_available > 0 or not self.balanced):
            self.negative_available -= 1
            return 0
        elif seizure_data.start_end[self.curr_seizure][0] <= timestamp < seizure_data.start_end[self.curr_seizure][1]:
            print("njdand")
            return 1
        
        if timestamp > seizure_data.start_end[self.curr_seizure][1] and self.curr_seizure < len(seizure_data.start_end):
            self.curr_seizure += 1

    def reset(self):
        self.curr_seizure = 1
        self.negative_available = -1