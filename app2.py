# app.py
"""
Interactive Caterer RAG Application.

- Initializes RAG pipeline once
- Accepts queries continuously
- Displays only the final answer
- Type 'exit', 'quit', or 'q' to stop
"""

from src.search import RAGSearch, SearchResponse


def print_final_response(response: SearchResponse) -> None:
    """Display only the final user-friendly answer."""

    print("\n" + "=" * 80)

    # Summary
    if response.summary:
        print(response.summary)

    # Caterer results
    for i, c in enumerate(response.caterers, start=1):

        print(f"\n{i}. {c.name}")

        print(f"   📍 Location       : {c.location}")
        print(f"   🌐 Region         : {c.region}")
        print(f"   🍽️ Specialization : {c.specialization}")
        print(f"   ⭐ Rating         : {c.rating}")
        print(f"   💰 Budget         : {c.budget_tier}")
        print(f"   👥 Capacity       : {c.guest_capacity}")

        if c.contact_phone:
            print(f"   📞 Phone          : {c.contact_phone}")

        if c.website:
            print(f"   🌍 Website        : {c.website}")

        if c.match_reason:
            print(f"   💡 Match          : {c.match_reason}")

    print("=" * 80)


def main():

    # ==========================================================
    # Initialize RAG pipeline ONCE
    # ==========================================================

    rag = RAGSearch(
        persist_dir="caterer_faiss_store",
        embedding_model="BAAI/bge-base-en-v1.5",
        llm_model="llama-3.3-70b-versatile",
    )

    # ==========================================================
    # Interactive query loop
    # ==========================================================

    while True:

        try:
            query = input("\n🔎 Ask about caterers: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Ignore empty queries
        if not query:
            continue

        # Exit commands
        if query.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye! 👋")
            break

        # ======================================================
        # Search
        # ======================================================

        try:

            response: SearchResponse = rag.search_and_summarize(
                query=query,
                top_k=3,
            )

            # ==================================================
            # Display ONLY final answer
            # ==================================================

            print_final_response(response)

        except Exception as e:

            print(
                f"\n❌ Sorry, something went wrong: {e}"
            )


if __name__ == "__main__":
    main()