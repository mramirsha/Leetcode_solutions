import collections
import itertools
import string
from collections import Counter


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    def remove_duplicate_letters(self, s: str) -> str:
        char_counts = Counter(s)
        taken = set()
        stack = []

        for c in s:
            if c not in taken:
                while stack and c < stack[-1] and char_counts[stack[-1]] > 0:
                    taken.remove(stack.pop())
                taken.add(c)
                stack.append(c)
                print(taken, stack)
            char_counts[c] -= 1

        return ''.join(stack)

    def two_sum(self, nums: list, target: int) -> list:
        num_hash_map = dict()
        for idx, each in enumerate(nums):
            num_hash_map[each] = idx

        for idx, element in enumerate(nums):
            if num_hash_map.get(target - element, False) and num_hash_map.get(target - element, False) != idx:
                return [idx, num_hash_map[target - element]]

    def add(self, l2, l1):
        result_list = list()
        tmp = 0
        for idx, ele_l2 in enumerate(l2):
            if idx < len(l1):
                res = ele_l2 + l1[idx] + tmp
                if res - 10 >= 0:
                    tmp = 1
                    result_list.append(res - 10)
                else:
                    result_list.append(res)
                    tmp = 0
            else:
                res = ele_l2 + tmp
                if res - 10 >= 0:
                    tmp = 1
                    result_list.append(res - 10)
                else:
                    result_list.append(res)
                    tmp = 0
        if tmp:
            result_list.append(tmp)
        return result_list

    def convert_llinked_list_to_list(self, list_: list):
        l = list()
        while list_:
            l.append(list_.val)
            list_ = list_.next
        return l

    def create_linked_list(self, l):
        linked = ListNode(l.pop)
        if l:
            linked.next = self.create_linked_list(l)
        return linked

    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        result_linked = ListNode(0)
        tmp = 0
        tmp_linked = result_linked
        while l1 or l2:
            l1_val = 0
            l2_val = 0
            if l1:
                l1_val = l1.val
                l1 = l1.next
            if l2:
                l2_val = l2.val
                l2 = l2.next
            res_val = l1_val + l2_val + tmp
            if res_val - 10 >= 0:
                tmp = 1
                result_val = res_val - 10
            else:
                result_val = res_val
                tmp = 0
            tmp_linked.next = ListNode(result_val)
            tmp_linked = tmp_linked.next
        if tmp:
            tmp_linked.next = ListNode(tmp)

        # print(result_linked)
        # return self.create_linked_list(result_linked)
        return result_linked.next

    def length_of_longest_substring(self, s: str) -> int:
        max_len = 0
        s_list = list(s)
        taken = list()
        max_taken = list()
        for e in s_list:
            # print(e, taken, max_len)
            if e not in taken:
                taken.append(e)
                if len(taken) > max_len:
                    max_len = len(taken)
                    max_taken = taken
            else:
                popped_letter = taken.pop(0)
                while e != popped_letter:
                    popped_letter = taken.pop(0)
                taken.append(e)

        print(max_taken)
        return max_len

    def binary_search(self, list_: list, num: int, left, right) -> int:
        if right >= left:
            mid = left + (right - left) // 2
            if num > list_[mid]:
                idx = self.binary_search(list_, num, mid + 1, right)
            elif num < list_[mid]:
                idx = self.binary_search(list_, num, left, mid - 1)
            else:
                return mid
        else:
            idx = 0 if right == -1 else right
        return idx

    def find_median_sorted_arrays(self, nums1: list, nums2: list) -> float:
        if not nums1:
            merged_list = nums2
        elif not nums2:
            merged_list = nums1
        else:
            if len(nums1) < len(nums2):
                nums1, nums2 = nums2, nums1

            left_idx = self.binary_search(nums1, nums2[0], 0, len(nums1) - 1)
            right_idx = self.binary_search(nums1, nums2[-1], 0, len(nums1) - 1)
            merged_list = nums1[:left_idx]
            flag = False
            if right_idx == left_idx == len(nums1) - 1:
                merged_list = nums1 + nums2
                flag = True
            while nums2 and not flag:
                if left_idx <= len(nums1) - 1:
                    if nums1[left_idx] < nums2[0]:
                        merged_list.append(nums1[left_idx])

                        left_idx += 1
                    elif nums2[0] < nums1[left_idx]:
                        merged_list.append(nums2.pop(0))

                    else:
                        merged_list.append(nums1[left_idx])
                        left_idx += 1
                        merged_list.append(nums2.pop(0))

                else:
                    merged_list.append(nums2.pop(0))
            if not flag:
                merged_list += nums1[left_idx:]
        middle_idx = (len(merged_list) / 2 - 1, (len(merged_list) / 2)) if len(merged_list) % 2 == 0 else (
            (len(merged_list) - 1) / 2, (len(merged_list) - 1) / 2)
        return (merged_list[int(middle_idx[0])] + merged_list[int(middle_idx[1])]) / 2

    def expand_around_center(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        if self.maxlen < right - left - 1:
            self.maxlen = right - left - 1
            self.start = left + 1

    def longest_palindrome(self, s: str) -> str:
        self.maxlen = 0
        self.start = 0
        if not s:
            return ""
        for i in range(len(s)):
            self.expandAroundCenter(s, i, i)
            self.expandAroundCenter(s, i, i + 1)
        return s[self.start: self.start + self.maxlen]

    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        rows = [''] * numRows
        cur_row, down = 0, False
        for c in s:
            rows[cur_row] += c
            if cur_row == 0 or cur_row == numRows - 1:
                down = not down
            cur_row += 1 if down else -1
        print(rows)
        return ''.join(rows)

    def reverse(self, x: int) -> int:
        l = list(str(x))
        new_list = list()
        if x < 0:
            new_list = [l.pop(0)]
        while l:
            new_list.append(l.pop())
        reversed_int = int("".join(new_list))
        if -(2 ** 31) <= reversed_int <= 2 ** 31 - 1:
            return reversed_int
        else:
            return 0

    def my_atoi(self, s: str) -> int:
        number = ''
        for c in s:
            if c == ' ' and not number:
                pass
            elif c.isnumeric():
                number += c
            elif (c == '+' or c == '-') and not number:
                number += c
            else:
                break
        print(number)
        number = 0 if number == '+' or number == '-' else number
        number = int(number) if number else 0
        if -(2 ** 31) > number:
            number = -(2 ** 31)
        elif number > 2 ** 31 - 1:
            number = 2 ** 31 - 1
        return number

    def is_palindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        revert_number = 0
        while x > revert_number:
            print(x, revert_number)
            revert_number = revert_number * 10 + x % 10
            x = int(10)
        print(x, revert_number)
        return x == revert_number or x == int(revert_number / 10)

    def is_match(self, s: str, p: str) -> bool:
        memo = {}

        def db(i, j):
            if (i, j) not in memo:
                if j == len(p):
                    ans = i == len(s)
                else:
                    first_match = i < len(s) and p[j] in [s[i], '.']
                    if j + 1 < len(p) and p[j + 1] == '*':
                        ans = db(i, j + 2) or first_match and db(i + 1, j)
                    else:
                        ans = first_match and db(i + 1, j + 1)
                memo[i, j] = ans
            return memo[i, j]

        return db(0, 0)

    def max_area(self, height: list) -> int:
        left = 0
        right = len(height) - 1
        max_width = len(height) - 1
        max_area = 0
        while max_width:
            if height[left] < height[right]:
                if max_area < height[left] * max_width:
                    max_area = height[left] * max_width
                left += 1
            else:
                if max_area < height[right] * max_width:
                    max_area = height[right] * max_width
                right -= 1
            max_width -= 1
        return max_area

    def int_to_roman(self, num: int) -> str:
        roman_symbol = {1: "I", 4: "IV", 5: "V", 9: "IX", 10: "X", 40: "XL", 50: "L", 90: "XC", 100: "C", 400: "CD",
                        500: "D", 900: "CM",
                        1000: "M"}
        res = ""
        while num:
            popped_item = roman_symbol.popitem()
            res += (num // popped_item[0]) * popped_item[1]
            num %= popped_item[0]
        return res

    def roman_to_int(self, s: str) -> int:
        roman_symbol = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400,
                        'D': 500, 'CM': 900, 'M': 1000}
        res, counter = 0, 0
        while counter < len(s):
            if counter + 1 < len(s) and roman_symbol.get(s[counter] + s[counter + 1]):
                res += roman_symbol.get(s[counter] + s[counter + 1])
                counter += 2
            else:
                res += roman_symbol.get(s[counter])
                counter += 1
        return res

    def longest_common_prefix(self, strs: list) -> str:
        if not strs:
            return ""
        common_pre = ""
        min_ = len(min(strs))
        for i in range(min_):
            tmp = ""
            for ele in strs:
                if tmp and tmp != ele[i]:
                    return common_pre
                tmp = ele[i]
            common_pre += tmp

        return common_pre

    def longest_valid_parentheses(self, s: str) -> int:
        ")(((((()())()()))()(()))("
        left, right = 0, 0
        max_len = 0
        for idx in range(len(s)):
            if s[idx] == ")":
                right += 1
            else:
                left += 1
            if left == right:
                max_len = max(max_len, left * 2)
            elif right > left:
                left, right = 0, 0
        left, right = 0, 0
        for idx in range(len(s)):
            if s[len(s) - idx - 1] == ")":
                right += 1
            else:
                left += 1
            if left == right:
                max_len = max(max_len, left * 2)
            elif left > right:
                left, right = 0, 0
        return max_len
        # stack = list()
        # final_list = list(s)
        # max_len = 0
        # for idx, each in enumerate(s):
        #     if each == ")" and stack:
        #         if stack[-1][1] == "(":
        #             popped = stack.pop()
        #             final_list[idx] = "2"
        #             final_list[popped[0]] = '0'
        #     else:
        #         stack.append((idx, each))
        # tmp = 0
        # for each in final_list:
        #     if each.isnumeric():
        #         tmp += int(each)
        #         if tmp > max_len:
        #             max_len = tmp
        #     else:
        #         tmp = 0
        # return max_len

    def three_sum(self, nums: list) -> list:
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res

    def three_sum_closest(self, nums: list, target: int) -> int:
        nums.sort()
        print(nums)
        len_num = len(nums)
        closest_distance = float("inf")
        closest_sum = 0
        for i in range(len_num - 2):
            print(closest_sum, closest_distance, i)
            l, r = i + 1, len_num - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s > target:
                    r -= 1
                    distance = s - target
                elif s < target:
                    l += 1
                    distance = target - s
                else:
                    return target
                if distance < closest_distance:
                    closest_distance = distance
                    closest_sum = s
                    # while l < r and nums[l] == nums[l + 1]:
                    #     l += 1
                    # while l < r and nums[r] == nums[r - 1]:
                    #     r -= 1
                # l += 1
                # r -= 1
        return closest_sum

    def dfs(self, nums, index, path, dic, res):
        if index >= len(nums):
            res.append(path)
            return res
        stringi = dic[nums[index]]
        for s in stringi:
            self.dfs(nums, index + 1, path + s, dic, res)
        return res

    def letter_combinations(self, digits: str) -> list:
        if not digits:
            return []
        dic = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res = []
        return self.dfs(digits, 0, "", dic, res)

    def two_sum(self, nums: list, target: int):
        res = []
        s = set()
        for idx, v in enumerate(nums):
            if len(res) == 0 or res[-1][1] != v:
                if target - v in s:
                    res.append([target - v, v])
            s.add(v)
        return res

    def k_sum(self, nums: list, target: int, k: int) -> list:
        nums.sort()
        res = []
        if not nums:
            return res
        if k == 2:
            return self.two_sum(nums, target)
        for idx, each in enumerate(nums):
            if idx == 0 or each != nums[idx - 1]:
                for subset in self.k_sum(nums[idx + 1:], target - each, k - 1):
                    res.append([each] + subset)
        return res

    def four_sum(self, nums: list, target: int) -> list:
        return self.k_sum(nums, target, 4)

    def remove_nth_from_end(self, head: ListNode, n: int) -> ListNode:
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        # print(slow.next.val)
        # print(fast.val)
        return head

    def is_valid(self, s: str) -> bool:
        stack = list()
        dic = {")": "(", "}": "{", "]": "["}
        for each in s:
            if each in {")", "}", "]"}:
                if not stack:
                    return False
                if stack.pop() != dic[each]:
                    return False
            else:
                stack.append(each)
        if stack:
            return False
        return True

    def merge_two_lists(self, list1: ListNode, list2: ListNode) -> ListNode:
        if list1 and list2:
            if list1.val > list2.val:
                res_list = ListNode(val=list2.val)
                list2 = list2.next
            else:
                res_list = ListNode(val=list1.val)
                list1 = list1.next
            tmp = res_list
            while list1 or list2:
                l = float('inf')
                r = float('inf')
                if list1:
                    l = list1.val
                if list2:
                    r = list2.val
                if l < r:
                    res_list.next = ListNode(val=l)
                    res_list = res_list.next
                    list1 = list1.next

                elif r < l:
                    res_list.next = ListNode(val=r)
                    res_list = res_list.next
                    list2 = list2.next
                else:
                    res_list.next = ListNode(val=l)
                    res_list = res_list.next
                    res_list.next = ListNode(val=r)
                    res_list = res_list.next
                    list2 = list2.next
                    list1 = list1.next
        else:
            if list1:
                tmp = list1
            else:
                tmp = list2
        return tmp

    def generate_parenthesis(self, N):
        if N == 0: return ['']
        ans = []
        for c in range(N):
            for left in self.generate_parenthesis(c):
                for right in self.generate_parenthesis(N - 1 - c):
                    ans.append('({}){}'.format(left, right))
        return ans

    def merge_k_lists(self, lists: list) -> ListNode:
        nodes = list()
        head = point = ListNode(0)
        for l in lists:
            while l:
                nodes.append(l.val)
                l = l.next
        for node in sorted(nodes):
            point.next = ListNode(node)
            point = point.next
        return head.next

    def swap_pairs(self, head: ListNode) -> ListNode:
        l = r = head
        new_head = point = ListNode(0)

        def next_(temp_l):
            if temp_l:
                temp_l = temp_l.next
            return temp_l

        while l or r:
            l = next_(l)
            if l:
                point.next = ListNode(l.val)
                point = point.next
            l = next_(l)
            if r:
                point.next = ListNode(r.val)
                point = point.next
            r = next_(next_(r))
        return new_head.next

    def reverse_k_group(self, head: ListNode, k: int) -> ListNode:
        dummy = jump = ListNode(-1)
        dummy.next = l = r = head

        while True:
            count = 0
            while r and count < k:
                count += 1
                r = r.next
            if count == k:
                pre, cur = r, l
                for _ in range(k):
                    print(cur.val, pre.val)
                    temp = cur.next
                    cur.next = pre
                    pre = cur
                    cur = temp
                jump.next = pre
                jump = l
                l = r
            else:
                return dummy.next

    def remove_duplicates(self, nums: list) -> int:
        k = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[k]:
                k += 1
                nums[k] = nums[i]
            print(nums, k)

        return k + 1

    def remove_element(self, nums: list, val: int) -> int:
        i = 0
        for idx in range(len(nums)):
            if nums[idx] != val:
                nums[i] = nums[idx]
                i += 1
        return i, nums

    def str_str(self, haystack: str, needle: str) -> list:
        if not needle:
            return []
        idx = -1
        idxes = list()
        i, j = 0, 0
        while j < len(haystack):
            if haystack[j] == needle[i]:
                if i == 0:
                    idx = j
                i += 1
            elif haystack[j] != needle[i] and i != 0:
                i = 0
                j = idx
            if i != 0 and i == len(needle):
                idxes.append(idx)
                i = 0
                j = idx
                idx = -1
            j += 1

        return idxes

    def divide(self, dividend: int, divisor: int) -> int:
        positive = (dividend < 0) is (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1
        if not positive:
            res = -res
        return min(max(-2147483648, res), 2147483647)

    def find_substring(self, s: str, words: list) -> list:
        n = len(s)
        k = len(words)
        word_len = len(words[0])
        substring_len = word_len * k
        words_set = collections.Counter(words)
        answer = list()

        def check(i: int):
            remaining = words_set.copy()
            word_count = 0
            for j in range(i, i + substring_len, word_len):
                sub = s[j: j + word_len]
                if remaining[sub] > 0:
                    remaining[sub] -= 1
                    word_count += 1
            return word_count == k

        for i in range(n - substring_len + 1):
            if check(i):
                answer.append(i)
        return answer

    def mergeSort(self, list_, l, r):
        mid = r - (r - l) // 2

    def next_permutation(self, nums: list) -> None:
        last_index = len(nums) - 1
        for i in range(len(nums)):
            print(i, last_index)
            if last_index - i >= 0:
                perv_idx = last_index - i - 1
                if nums[last_index - i] > nums[perv_idx]:
                    if i != 0:
                        flag = True
                        while i > -1:
                            if nums[last_index - i] <= nums[perv_idx]:
                                i += 1
                                flag = False
                                break
                            else:
                                i -= 1
                        if flag:
                            i = 0

                    print(i, perv_idx)
                    nums[last_index - i], nums[perv_idx] = nums[perv_idx], nums[last_index - i]
                    nums[perv_idx + 1:] = sorted(nums[perv_idx + 1:])
                    break
            else:
                print('sorted')
                nums.sort()
        print(nums)

    def search(self, nums: list, target: int) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = r - (r - l) // 2 - 1
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        rot = l
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = r - (r - l) // 2
            realmid = (mid + rot) % len(nums)
            if nums[realmid] == target:
                return realmid
            elif nums[realmid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return -1

    def search_range(self, nums: list, target: int) -> list:
        if not nums:
            return [-1, -1]

        def find_starting_index(nums, target):
            index = -1
            low, high = 0, len(nums) - 1

            while low <= high:
                mid = high - (high - low) // 2

                if nums[mid] == target:
                    index = mid
                    high = mid - 1
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1

            return index

        def find_ending_index(nums, target):
            index = -1
            low, high = 0, len(nums) - 1

            while low <= high:
                mid = high - (high - low) // 2

                if nums[mid] == target:
                    index = mid
                    low = mid + 1
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
            return index

        return [find_starting_index(nums, target), find_ending_index(nums, target)]

    def search_insert(self, nums: list, target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = r - (r - l) // 2
            if nums[mid] == target:
                index = mid
                break
            if nums[mid] > target:
                r = mid - 1
                index = mid
            else:
                l = mid + 1
                index = mid + 1
        return index

    def is_valid_sudoku(self, board: list) -> bool:

        def unit_is_valid(unit: list):
            numbers = [each for each in unit if each != '.']
            return len(set(numbers)) == len(numbers)

        def check_row_is_valid():
            for i in range(9):
                if not unit_is_valid(board[i]):
                    return False
            return True

        def check_column_is_valid():
            for i in range(9):
                column = list()
                for j in range(9):
                    column.append(board[j][i])
                if not unit_is_valid(column):
                    return False
            return True

        def check_square_is_valid():
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    square = list()
                    for row in range(0 + i, 3 + i):
                        for column in range(0 + j, 3 + j):
                            square.append(board[column][row])
                    if not unit_is_valid(square):
                        return False
            return True

        return check_square_is_valid() and check_row_is_valid() and check_column_is_valid()

    def solve_sudoku(self, board: list) -> None:

        def find_unassigned():
            for row in range(9):
                for col in range(9):
                    if board[row][col] == '.':
                        return row, col
            return -1, -1

        def is_valid_row(row, num):
            for i in range(9):
                if board[row][i] == num:
                    return False
            return True

        def is_valid_col(col, num):
            for i in range(9):
                if board[i][col] == num:
                    return False
            return True

        def is_valid_square(row, col, num):
            for r in range(row, row + 3):
                for c in range(col, col + 3):
                    if board[r][c] == num:
                        return False
            return True

        def is_safe(row, col, num):
            box_rows = row - row % 3
            box_col = col - col % 3
            return is_valid_col(col, num) and is_valid_row(row, num) and is_valid_square(box_rows, box_col, num)

        def solve():
            row, col = find_unassigned()
            if row == -1:
                return True
            for num in range(1, 10):
                if is_safe(row, col, str(num)):
                    board[row][col] = str(num)
                    if solve():
                        return True
                    board[row][col] = '.'
            return False

        solve()

    def count_and_say(self, n: int) -> str:
        result = '1'
        for _ in range(n - 1):
            v = ''
            print(result)
            for d, g in itertools.groupby(result):
                v += f"{len(list(g))}{d}"
            result = v
        return result


import collections
import itertools
import string
from collections import Counter


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    def remove_duplicate_letters(self, s: str) -> str:
        char_counts = Counter(s)
        taken = set()
        stack = []

        for c in s:
            if c not in taken:
                while stack and c < stack[-1] and char_counts[stack[-1]] > 0:
                    taken.remove(stack.pop())
                taken.add(c)
                stack.append(c)
                print(taken, stack)
            char_counts[c] -= 1

        return ''.join(stack)

    def two_sum(self, nums: list, target: int) -> list:
        num_hash_map = dict()
        for idx, each in enumerate(nums):
            num_hash_map[each] = idx

        for idx, element in enumerate(nums):
            if num_hash_map.get(target - element, False) and num_hash_map.get(target - element, False) != idx:
                return [idx, num_hash_map[target - element]]

    def add(self, l2, l1):
        result_list = list()
        tmp = 0
        for idx, ele_l2 in enumerate(l2):
            if idx < len(l1):
                res = ele_l2 + l1[idx] + tmp
                if res - 10 >= 0:
                    tmp = 1
                    result_list.append(res - 10)
                else:
                    result_list.append(res)
                    tmp = 0
            else:
                res = ele_l2 + tmp
                if res - 10 >= 0:
                    tmp = 1
                    result_list.append(res - 10)
                else:
                    result_list.append(res)
                    tmp = 0
        if tmp:
            result_list.append(tmp)
        return result_list

    def convert_llinked_list_to_list(self, list_: list):
        l = list()
        while list_:
            l.append(list_.val)
            list_ = list_.next
        return l

    def create_linked_list(self, l):
        linked = ListNode(l.pop)
        if l:
            linked.next = self.create_linked_list(l)
        return linked

    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        result_linked = ListNode(0)
        tmp = 0
        tmp_linked = result_linked
        while l1 or l2:
            l1_val = 0
            l2_val = 0
            if l1:
                l1_val = l1.val
                l1 = l1.next
            if l2:
                l2_val = l2.val
                l2 = l2.next
            res_val = l1_val + l2_val + tmp
            if res_val - 10 >= 0:
                tmp = 1
                result_val = res_val - 10
            else:
                result_val = res_val
                tmp = 0
            tmp_linked.next = ListNode(result_val)
            tmp_linked = tmp_linked.next
        if tmp:
            tmp_linked.next = ListNode(tmp)

        # print(result_linked)
        # return self.create_linked_list(result_linked)
        return result_linked.next

    def length_of_longest_substring(self, s: str) -> int:
        max_len = 0
        s_list = list(s)
        taken = list()
        max_taken = list()
        for e in s_list:
            # print(e, taken, max_len)
            if e not in taken:
                taken.append(e)
                if len(taken) > max_len:
                    max_len = len(taken)
                    max_taken = taken
            else:
                popped_letter = taken.pop(0)
                while e != popped_letter:
                    popped_letter = taken.pop(0)
                taken.append(e)

        print(max_taken)
        return max_len

    def binary_search(self, list_: list, num: int, left, right) -> int:
        if right >= left:
            mid = left + (right - left) // 2
            if num > list_[mid]:
                idx = self.binary_search(list_, num, mid + 1, right)
            elif num < list_[mid]:
                idx = self.binary_search(list_, num, left, mid - 1)
            else:
                return mid
        else:
            idx = 0 if right == -1 else right
        return idx

    def find_median_sorted_arrays(self, nums1: list, nums2: list) -> float:
        if not nums1:
            merged_list = nums2
        elif not nums2:
            merged_list = nums1
        else:
            if len(nums1) < len(nums2):
                nums1, nums2 = nums2, nums1

            left_idx = self.binary_search(nums1, nums2[0], 0, len(nums1) - 1)
            right_idx = self.binary_search(nums1, nums2[-1], 0, len(nums1) - 1)
            merged_list = nums1[:left_idx]
            flag = False
            if right_idx == left_idx == len(nums1) - 1:
                merged_list = nums1 + nums2
                flag = True
            while nums2 and not flag:
                if left_idx <= len(nums1) - 1:
                    if nums1[left_idx] < nums2[0]:
                        merged_list.append(nums1[left_idx])

                        left_idx += 1
                    elif nums2[0] < nums1[left_idx]:
                        merged_list.append(nums2.pop(0))

                    else:
                        merged_list.append(nums1[left_idx])
                        left_idx += 1
                        merged_list.append(nums2.pop(0))

                else:
                    merged_list.append(nums2.pop(0))
            if not flag:
                merged_list += nums1[left_idx:]
        middle_idx = (len(merged_list) / 2 - 1, (len(merged_list) / 2)) if len(merged_list) % 2 == 0 else (
            (len(merged_list) - 1) / 2, (len(merged_list) - 1) / 2)
        return (merged_list[int(middle_idx[0])] + merged_list[int(middle_idx[1])]) / 2

    def expand_around_center(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        if self.maxlen < right - left - 1:
            self.maxlen = right - left - 1
            self.start = left + 1

    def longest_palindrome(self, s: str) -> str:
        self.maxlen = 0
        self.start = 0
        if not s:
            return ""
        for i in range(len(s)):
            self.expandAroundCenter(s, i, i)
            self.expandAroundCenter(s, i, i + 1)
        return s[self.start: self.start + self.maxlen]

    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        rows = [''] * numRows
        cur_row, down = 0, False
        for c in s:
            rows[cur_row] += c
            if cur_row == 0 or cur_row == numRows - 1:
                down = not down
            cur_row += 1 if down else -1
        print(rows)
        return ''.join(rows)

    def reverse(self, x: int) -> int:
        l = list(str(x))
        new_list = list()
        if x < 0:
            new_list = [l.pop(0)]
        while l:
            new_list.append(l.pop())
        reversed_int = int("".join(new_list))
        if -(2 ** 31) <= reversed_int <= 2 ** 31 - 1:
            return reversed_int
        else:
            return 0

    def my_atoi(self, s: str) -> int:
        number = ''
        for c in s:
            if c == ' ' and not number:
                pass
            elif c.isnumeric():
                number += c
            elif (c == '+' or c == '-') and not number:
                number += c
            else:
                break
        print(number)
        number = 0 if number == '+' or number == '-' else number
        number = int(number) if number else 0
        if -(2 ** 31) > number:
            number = -(2 ** 31)
        elif number > 2 ** 31 - 1:
            number = 2 ** 31 - 1
        return number

    def is_palindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        revert_number = 0
        while x > revert_number:
            print(x, revert_number)
            revert_number = revert_number * 10 + x % 10
            x = int(10)
        print(x, revert_number)
        return x == revert_number or x == int(revert_number / 10)

    def is_match(self, s: str, p: str) -> bool:
        memo = {}

        def db(i, j):
            if (i, j) not in memo:
                if j == len(p):
                    ans = i == len(s)
                else:
                    first_match = i < len(s) and p[j] in [s[i], '.']
                    if j + 1 < len(p) and p[j + 1] == '*':
                        ans = db(i, j + 2) or first_match and db(i + 1, j)
                    else:
                        ans = first_match and db(i + 1, j + 1)
                memo[i, j] = ans
            return memo[i, j]

        return db(0, 0)

    def max_area(self, height: list) -> int:
        left = 0
        right = len(height) - 1
        max_width = len(height) - 1
        max_area = 0
        while max_width:
            if height[left] < height[right]:
                if max_area < height[left] * max_width:
                    max_area = height[left] * max_width
                left += 1
            else:
                if max_area < height[right] * max_width:
                    max_area = height[right] * max_width
                right -= 1
            max_width -= 1
        return max_area

    def int_to_roman(self, num: int) -> str:
        roman_symbol = {1: "I", 4: "IV", 5: "V", 9: "IX", 10: "X", 40: "XL", 50: "L", 90: "XC", 100: "C", 400: "CD",
                        500: "D", 900: "CM",
                        1000: "M"}
        res = ""
        while num:
            popped_item = roman_symbol.popitem()
            res += (num // popped_item[0]) * popped_item[1]
            num %= popped_item[0]
        return res

    def roman_to_int(self, s: str) -> int:
        roman_symbol = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400,
                        'D': 500, 'CM': 900, 'M': 1000}
        res, counter = 0, 0
        while counter < len(s):
            if counter + 1 < len(s) and roman_symbol.get(s[counter] + s[counter + 1]):
                res += roman_symbol.get(s[counter] + s[counter + 1])
                counter += 2
            else:
                res += roman_symbol.get(s[counter])
                counter += 1
        return res

    def longest_common_prefix(self, strs: list) -> str:
        if not strs:
            return ""
        common_pre = ""
        min_ = len(min(strs))
        for i in range(min_):
            tmp = ""
            for ele in strs:
                if tmp and tmp != ele[i]:
                    return common_pre
                tmp = ele[i]
            common_pre += tmp

        return common_pre

    def longest_valid_parentheses(self, s: str) -> int:
        ")(((((()())()()))()(()))("
        left, right = 0, 0
        max_len = 0
        for idx in range(len(s)):
            if s[idx] == ")":
                right += 1
            else:
                left += 1
            if left == right:
                max_len = max(max_len, left * 2)
            elif right > left:
                left, right = 0, 0
        left, right = 0, 0
        for idx in range(len(s)):
            if s[len(s) - idx - 1] == ")":
                right += 1
            else:
                left += 1
            if left == right:
                max_len = max(max_len, left * 2)
            elif left > right:
                left, right = 0, 0
        return max_len
        # stack = list()
        # final_list = list(s)
        # max_len = 0
        # for idx, each in enumerate(s):
        #     if each == ")" and stack:
        #         if stack[-1][1] == "(":
        #             popped = stack.pop()
        #             final_list[idx] = "2"
        #             final_list[popped[0]] = '0'
        #     else:
        #         stack.append((idx, each))
        # tmp = 0
        # for each in final_list:
        #     if each.isnumeric():
        #         tmp += int(each)
        #         if tmp > max_len:
        #             max_len = tmp
        #     else:
        #         tmp = 0
        # return max_len

    def three_sum(self, nums: list) -> list:
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res

    def three_sum_closest(self, nums: list, target: int) -> int:
        nums.sort()
        print(nums)
        len_num = len(nums)
        closest_distance = float("inf")
        closest_sum = 0
        for i in range(len_num - 2):
            print(closest_sum, closest_distance, i)
            l, r = i + 1, len_num - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s > target:
                    r -= 1
                    distance = s - target
                elif s < target:
                    l += 1
                    distance = target - s
                else:
                    return target
                if distance < closest_distance:
                    closest_distance = distance
                    closest_sum = s
                    # while l < r and nums[l] == nums[l + 1]:
                    #     l += 1
                    # while l < r and nums[r] == nums[r - 1]:
                    #     r -= 1
                # l += 1
                # r -= 1
        return closest_sum

    def dfs(self, nums, index, path, dic, res):
        if index >= len(nums):
            res.append(path)
            return res
        stringi = dic[nums[index]]
        for s in stringi:
            self.dfs(nums, index + 1, path + s, dic, res)
        return res

    def letter_combinations(self, digits: str) -> list:
        if not digits:
            return []
        dic = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res = []
        return self.dfs(digits, 0, "", dic, res)

    def two_sum(self, nums: list, target: int):
        res = []
        s = set()
        for idx, v in enumerate(nums):
            if len(res) == 0 or res[-1][1] != v:
                if target - v in s:
                    res.append([target - v, v])
            s.add(v)
        return res

    def k_sum(self, nums: list, target: int, k: int) -> list:
        nums.sort()
        res = []
        if not nums:
            return res
        if k == 2:
            return self.two_sum(nums, target)
        for idx, each in enumerate(nums):
            if idx == 0 or each != nums[idx - 1]:
                for subset in self.k_sum(nums[idx + 1:], target - each, k - 1):
                    res.append([each] + subset)
        return res

    def four_sum(self, nums: list, target: int) -> list:
        return self.k_sum(nums, target, 4)

    def remove_nth_from_end(self, head: ListNode, n: int) -> ListNode:
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        # print(slow.next.val)
        # print(fast.val)
        return head

    def is_valid(self, s: str) -> bool:
        stack = list()
        dic = {")": "(", "}": "{", "]": "["}
        for each in s:
            if each in {")", "}", "]"}:
                if not stack:
                    return False
                if stack.pop() != dic[each]:
                    return False
            else:
                stack.append(each)
        if stack:
            return False
        return True

    def merge_two_lists(self, list1: ListNode, list2: ListNode) -> ListNode:
        if list1 and list2:
            if list1.val > list2.val:
                res_list = ListNode(val=list2.val)
                list2 = list2.next
            else:
                res_list = ListNode(val=list1.val)
                list1 = list1.next
            tmp = res_list
            while list1 or list2:
                l = float('inf')
                r = float('inf')
                if list1:
                    l = list1.val
                if list2:
                    r = list2.val
                if l < r:
                    res_list.next = ListNode(val=l)
                    res_list = res_list.next
                    list1 = list1.next

                elif r < l:
                    res_list.next = ListNode(val=r)
                    res_list = res_list.next
                    list2 = list2.next
                else:
                    res_list.next = ListNode(val=l)
                    res_list = res_list.next
                    res_list.next = ListNode(val=r)
                    res_list = res_list.next
                    list2 = list2.next
                    list1 = list1.next
        else:
            if list1:
                tmp = list1
            else:
                tmp = list2
        return tmp

    def generate_parenthesis(self, N):
        if N == 0: return ['']
        ans = []
        for c in range(N):
            for left in self.generate_parenthesis(c):
                for right in self.generate_parenthesis(N - 1 - c):
                    ans.append('({}){}'.format(left, right))
        return ans

    def merge_k_lists(self, lists: list) -> ListNode:
        nodes = list()
        head = point = ListNode(0)
        for l in lists:
            while l:
                nodes.append(l.val)
                l = l.next
        for node in sorted(nodes):
            point.next = ListNode(node)
            point = point.next
        return head.next

    def swap_pairs(self, head: ListNode) -> ListNode:
        l = r = head
        new_head = point = ListNode(0)

        def next_(temp_l):
            if temp_l:
                temp_l = temp_l.next
            return temp_l

        while l or r:
            l = next_(l)
            if l:
                point.next = ListNode(l.val)
                point = point.next
            l = next_(l)
            if r:
                point.next = ListNode(r.val)
                point = point.next
            r = next_(next_(r))
        return new_head.next

    def reverse_k_group(self, head: ListNode, k: int) -> ListNode:
        dummy = jump = ListNode(-1)
        dummy.next = l = r = head

        while True:
            count = 0
            while r and count < k:
                count += 1
                r = r.next
            if count == k:
                pre, cur = r, l
                for _ in range(k):
                    print(cur.val, pre.val)
                    temp = cur.next
                    cur.next = pre
                    pre = cur
                    cur = temp
                jump.next = pre
                jump = l
                l = r
            else:
                return dummy.next

    def remove_duplicates(self, nums: list) -> int:
        k = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[k]:
                k += 1
                nums[k] = nums[i]
            print(nums, k)

        return k + 1

    def remove_element(self, nums: list, val: int) -> int:
        i = 0
        for idx in range(len(nums)):
            if nums[idx] != val:
                nums[i] = nums[idx]
                i += 1
        return i, nums

    def str_str(self, haystack: str, needle: str) -> list:
        if not needle:
            return []
        idx = -1
        idxes = list()
        i, j = 0, 0
        while j < len(haystack):
            if haystack[j] == needle[i]:
                if i == 0:
                    idx = j
                i += 1
            elif haystack[j] != needle[i] and i != 0:
                i = 0
                j = idx
            if i != 0 and i == len(needle):
                idxes.append(idx)
                i = 0
                j = idx
                idx = -1
            j += 1

        return idxes

    def divide(self, dividend: int, divisor: int) -> int:
        positive = (dividend < 0) is (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1
        if not positive:
            res = -res
        return min(max(-2147483648, res), 2147483647)

    def find_substring(self, s: str, words: list) -> list:
        n = len(s)
        k = len(words)
        word_len = len(words[0])
        substring_len = word_len * k
        words_set = collections.Counter(words)
        answer = list()

        def check(i: int):
            remaining = words_set.copy()
            word_count = 0
            for j in range(i, i + substring_len, word_len):
                sub = s[j: j + word_len]
                if remaining[sub] > 0:
                    remaining[sub] -= 1
                    word_count += 1
            return word_count == k

        for i in range(n - substring_len + 1):
            if check(i):
                answer.append(i)
        return answer

    def mergeSort(self, list_, l, r):
        mid = r - (r - l) // 2

    def next_permutation(self, nums: list) -> None:
        last_index = len(nums) - 1
        for i in range(len(nums)):
            print(i, last_index)
            if last_index - i >= 0:
                perv_idx = last_index - i - 1
                if nums[last_index - i] > nums[perv_idx]:
                    if i != 0:
                        flag = True
                        while i > -1:
                            if nums[last_index - i] <= nums[perv_idx]:
                                i += 1
                                flag = False
                                break
                            else:
                                i -= 1
                        if flag:
                            i = 0

                    print(i, perv_idx)
                    nums[last_index - i], nums[perv_idx] = nums[perv_idx], nums[last_index - i]
                    nums[perv_idx + 1:] = sorted(nums[perv_idx + 1:])
                    break
            else:
                print('sorted')
                nums.sort()
        print(nums)

    def search(self, nums: list, target: int) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = r - (r - l) // 2 - 1
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        rot = l
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = r - (r - l) // 2
            realmid = (mid + rot) % len(nums)
            if nums[realmid] == target:
                return realmid
            elif nums[realmid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return -1

    def search_range(self, nums: list, target: int) -> list:
        if not nums:
            return [-1, -1]

        def find_starting_index(nums, target):
            index = -1
            low, high = 0, len(nums) - 1

            while low <= high:
                mid = high - (high - low) // 2

                if nums[mid] == target:
                    index = mid
                    high = mid - 1
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1

            return index

        def find_ending_index(nums, target):
            index = -1
            low, high = 0, len(nums) - 1

            while low <= high:
                mid = high - (high - low) // 2

                if nums[mid] == target:
                    index = mid
                    low = mid + 1
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
            return index

        return [find_starting_index(nums, target), find_ending_index(nums, target)]

    def search_insert(self, nums: list, target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = r - (r - l) // 2
            if nums[mid] == target:
                index = mid
                break
            if nums[mid] > target:
                r = mid - 1
                index = mid
            else:
                l = mid + 1
                index = mid + 1
        return index

    def is_valid_sudoku(self, board: list) -> bool:

        def unit_is_valid(unit: list):
            numbers = [each for each in unit if each != '.']
            return len(set(numbers)) == len(numbers)

        def check_row_is_valid():
            for i in range(9):
                if not unit_is_valid(board[i]):
                    return False
            return True

        def check_column_is_valid():
            for i in range(9):
                column = list()
                for j in range(9):
                    column.append(board[j][i])
                if not unit_is_valid(column):
                    return False
            return True

        def check_square_is_valid():
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    square = list()
                    for row in range(0 + i, 3 + i):
                        for column in range(0 + j, 3 + j):
                            square.append(board[column][row])
                    if not unit_is_valid(square):
                        return False
            return True

        return check_square_is_valid() and check_row_is_valid() and check_column_is_valid()

    def solve_sudoku(self, board: list) -> None:

        def find_unassigned():
            for row in range(9):
                for col in range(9):
                    if board[row][col] == '.':
                        return row, col
            return -1, -1

        def is_valid_row(row, num):
            for i in range(9):
                if board[row][i] == num:
                    return False
            return True

        def is_valid_col(col, num):
            for i in range(9):
                if board[i][col] == num:
                    return False
            return True

        def is_valid_square(row, col, num):
            for r in range(row, row + 3):
                for c in range(col, col + 3):
                    if board[r][c] == num:
                        return False
            return True

        def is_safe(row, col, num):
            box_rows = row - row % 3
            box_col = col - col % 3
            return is_valid_col(col, num) and is_valid_row(row, num) and is_valid_square(box_rows, box_col, num)

        def solve():
            row, col = find_unassigned()
            if row == -1:
                return True
            for num in range(1, 10):
                if is_safe(row, col, str(num)):
                    board[row][col] = str(num)
                    if solve():
                        return True
                    board[row][col] = '.'
            return False

        solve()

    def count_and_say(self, n: int) -> str:
        result = '1'
        for _ in range(n - 1):
            v = ''
            print(result)
            for d, g in itertools.groupby(result):
                v += f"{len(list(g))}{d}"
            result = v
        return result
