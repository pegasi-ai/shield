import type { Answers, QuestionCollection } from 'inquirer' with { 'resolution-mode': 'import' };

/** Load the ESM prompt package only when an interactive prompt is needed. */
export async function prompt<T extends Answers = Answers>(
  questions: QuestionCollection<T>,
  initialAnswers?: Partial<T>
): Promise<T> {
  const { default: inquirer } = await import('inquirer');
  return inquirer.prompt<T>(questions, initialAnswers);
}
