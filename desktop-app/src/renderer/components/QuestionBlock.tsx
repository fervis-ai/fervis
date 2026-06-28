export function QuestionBlock({ question }: { readonly question: string }) {
  return (
    <section className="question-block">
      <div className="section-label">
        Question <span>§1</span>
      </div>
      <p>{question}</p>
    </section>
  );
}
