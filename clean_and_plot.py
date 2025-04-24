import pandas as pd
import matplotlib.pyplot as plt
import re

# dataset path
output_dataset_path = 'C:\\Users\\vimuk\\PycharmProjects\\dataset_gen\\generated_dataset_reward\\pair_gsm8k_v2.csv'

# Loading dataset
df = pd.read_csv(output_dataset_path, encoding='ISO-8859-1')

# Printing original row count
original_row_count = len(df)
print(f"Original rows in dataset: {original_row_count}")

# Removing rows with null chosen_response
df = df.dropna(subset=['chosen_response'])

# Defining all required tags
required_tags = ['<goal_detector>', '</goal_detector>',
                 '<plan_generator>', '</plan_generator>',
                 '<projector>', '</projector>',
                 '<executer>', '</executer>']

# Function to check if all required tags are present in a response
def has_all_tags(text):
    return all(tag in text for tag in required_tags)

# Filtering rows with all tags present
df_cleaned = df[df['chosen_response'].apply(has_all_tags)]

# Calculating removed rows
cleaned_row_count = len(df_cleaned)
removed_row_count = original_row_count - cleaned_row_count

print(f"Rows removed (null or missing tags): {removed_row_count}")
print(f"Rows retained after cleaning: {cleaned_row_count}")

# Function to count tag occurrences
def count_tag(tag, text_series):
    return text_series.str.count(re.escape(tag)).sum()

# Count each tag
tag_counts = {tag: count_tag(tag, df_cleaned['chosen_response']) for tag in ['<goal_detector>', '<plan_generator>', '<projector>', '<executer>']}

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(tag_counts.keys(), tag_counts.values())
plt.title('Count of Reasoning Tags in Cleaned Dataset')
plt.ylabel('Count')
plt.xlabel('Tag')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Saving the cleaned dataset
df_cleaned.to_csv('C:\\Users\\vimuk\\PycharmProjects\\dataset_gen\\generated_dataset_reward\\pair_gsm8k_v2.csv', index=False)
