"""
Course Recommendation System — Transparent Glass Theme
======================================================

Install required libraries:
    pip install pandas scikit-learn

Run:
    python course_recommendation_system.py

Dataset:
    udemy_courses.csv

Required CSV columns:
    course_title
    subject

Optional CSV columns:
    num_subscribers
    num_reviews

Features:
    - Dark transparent-style interface
    - Matching buttons with hover effects and shadows
    - Course search
    - Course similarity recommendations
    - Popular course recommendations
    - CSV dataset loading
    - Dataset statistics
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# COLORS — TRANSPARENT GLASS-STYLE THEME
# ============================================================

BG = "#202632"
BG_ALT = "#293241"

CARD = "#303A4B"
CARD_HOVER = "#39465A"

BUTTON = "#5267A8"
BUTTON_HOVER = "#647BC1"

BUTTON_SECONDARY = "#3B465A"
BUTTON_SECONDARY_HOVER = "#4A5870"

TEXT = "#F4F6FB"
MUTED = "#B4BECE"

BORDER = "#465267"
SHADOW = "#111722"

ACCENT = "#A9B9FF"
GREEN = "#79D6B0"

FONT = "Segoe UI"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """Convert text to a normalized lowercase string."""
    if pd.isna(value):
        return ""

    return " ".join(
        str(value).lower().replace("_", " ").split()
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

class CourseRecommendationApp:

    def __init__(self, root):

        self.root = root

        # Window configuration
        self.root.title(
            "ZaBir | Course Finder "
        )

        self.root.geometry("1080x800")
        self.root.minsize(850, 650)
        self.root.configure(bg=BG)

        # Slight whole-window transparency.
        # Tkinter does not support separate transparency for widgets.
        try:
            self.root.attributes("-alpha", 0.96)
        except tk.TclError:
            pass

        # Application data
        self.data = pd.DataFrame()
        self.similarity = None
        self.course_titles = []
        self.csv_path = None

        # Build interface
        self._configure_styles()
        self._build_ui()

        # Automatically load the default dataset if available
        default_csv = (
            Path(__file__).resolve().parent / "Courses.csv"
        )

        if default_csv.exists():
            self.load_dataset(str(default_csv))
        else:
            self._set_status(
                "Load Courses.csv to begin."
            )
            self._show_empty_state()

    # ========================================================
    # STYLE CONFIGURATION
    # ========================================================

    def _configure_styles(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Glass.TCombobox",
            fieldbackground=CARD,
            background=CARD,
            foreground=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            arrowcolor=ACCENT,
            padding=10,
            font=(FONT, 10),
        )

        style.map(
            "Glass.TCombobox",
            fieldbackground=[
                ("readonly", CARD),
                ("focus", CARD),
            ],
            foreground=[
                ("readonly", TEXT),
                ("focus", TEXT),
            ],
            selectbackground=[
                ("readonly", BUTTON),
            ],
            selectforeground=[
                ("readonly", TEXT),
            ],
        )

        style.configure(
            "Glass.Vertical.TScrollbar",
            background=BUTTON_SECONDARY,
            troughcolor=BG,
            bordercolor=BG,
            arrowcolor=TEXT,
        )

    # ========================================================
    # BUILD USER INTERFACE
    # ========================================================

    def _build_ui(self):

        # Main container
        outer = tk.Frame(
            self.root,
            bg=BG
        )

        outer.pack(
            fill="both",
            expand=True,
            padx=32,
            pady=24
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = tk.Frame(
            outer,
            bg=BG
        )

        header.pack(
            fill="x",
            pady=(0, 20)
        )

        tk.Label(
            header,
            text="📚 ZaBir HaSan",
            bg=BG,
            fg=ACCENT,
            font=(FONT, 20, "bold"),
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Find your next great course.",
            bg=BG,
            fg=TEXT,
            font=(FONT, 27, "bold"),
        ).pack(
            anchor="w",
            pady=(5, 2)
        )

        tk.Label(
            header,
            text=(
                "Personalized suggestions from course similarity "
                "and learner popularity."
            ),
            bg=BG,
            fg=MUTED,
            font=(FONT, 10),
        ).pack(anchor="w")

        # ----------------------------------------------------
        # STATISTICS CARDS
        # ----------------------------------------------------

        metrics = tk.Frame(
            outer,
            bg=BG
        )

        metrics.pack(
            fill="x",
            pady=(0, 18)
        )

        self.course_count_var = tk.StringVar(
            value="—"
        )

        self.subject_count_var = tk.StringVar(
            value="—"
        )

        self.dataset_var = tk.StringVar(
            value="Not loaded"
        )

        self._metric_card(
            metrics,
            "COURSES IN DATASET",
            self.course_count_var,
            0
        )

        self._metric_card(
            metrics,
            "SUBJECT AREAS",
            self.subject_count_var,
            1
        )

        self._metric_card(
            metrics,
            "DATA SOURCE",
            self.dataset_var,
            2
        )

        # ----------------------------------------------------
        # SEARCH CARD WITH SHADOW
        # ----------------------------------------------------

        search_shadow = tk.Frame(
            outer,
            bg=SHADOW
        )

        search_shadow.pack(
            fill="x",
            pady=(0, 16)
        )

        search_card = tk.Frame(
            search_shadow,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        search_card.pack(
            fill="x",
            padx=(0, 3),
            pady=(0, 3)
        )

        inner = tk.Frame(
            search_card,
            bg=CARD
        )

        inner.pack(
            fill="x",
            padx=22,
            pady=20
        )

        tk.Label(
            inner,
            text="CHOOSE A COURSE",
            bg=CARD,
            fg=MUTED,
            font=(FONT, 9, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        # Search row
        search_row = tk.Frame(
            inner,
            bg=CARD
        )

        search_row.pack(fill="x")

        self.course_var = tk.StringVar()

        self.search_box = ttk.Combobox(
            search_row,
            textvariable=self.course_var,
            state="normal",
            style="Glass.TCombobox",
            font=(FONT, 10),
        )

        self.search_box.pack(
            side="left",
            fill="x",
            expand=True,
            ipady=2
        )

        self.search_box.bind(
            "<KeyRelease>",
            self._filter_course_choices
        )

        self.search_box.bind(
            "<Return>",
            lambda _event: self.recommend()
        )

        # Recommend button
        self._shadow_button(
            search_row,
            "✦  Recommend courses",
            self.recommend,
            primary=True
        ).pack(
            side="left",
            padx=(12, 0)
        )

        # Popular courses button
        self._shadow_button(
            search_row,
            "Popular courses",
            self.show_popular,
            primary=False
        ).pack(
            side="left",
            padx=(10, 0)
        )

        # Load dataset button
        self._shadow_button(
            search_row,
            "Load dataset",
            self.choose_dataset,
            primary=False
        ).pack(
            side="left",
            padx=(10, 0)
        )

        # ----------------------------------------------------
        # RESULTS HEADER
        # ----------------------------------------------------

        results_heading = tk.Frame(
            outer,
            bg=BG
        )

        results_heading.pack(
            fill="x",
            pady=(2, 10)
        )

        self.results_title = tk.Label(
            results_heading,
            text="Popular courses",
            bg=BG,
            fg=TEXT,
            font=(FONT, 15, "bold"),
        )

        self.results_title.pack(
            side="left"
        )

        self.results_note = tk.Label(
            results_heading,
            text="Ranked by subscriber and review activity",
            bg=BG,
            fg=MUTED,
            font=(FONT, 9),
        )

        self.results_note.pack(
            side="right"
        )

        # ----------------------------------------------------
        # RESULTS CARD WITH SHADOW
        # ----------------------------------------------------

        results_shadow = tk.Frame(
            outer,
            bg=SHADOW
        )

        results_shadow.pack(
            fill="both",
            expand=True
        )

        self.results_card = tk.Frame(
            results_shadow,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        self.results_card.pack(
            fill="both",
            expand=True,
            padx=(0, 3),
            pady=(0, 3)
        )

        self.results_body = tk.Frame(
            self.results_card,
            bg=CARD
        )

        self.results_body.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=14
        )

        # ----------------------------------------------------
        # STATUS BAR
        # ----------------------------------------------------

        self.status_var = tk.StringVar(
            value="Ready"
        )

        status = tk.Frame(
            outer,
            bg=BG
        )

        status.pack(
            fill="x",
            pady=(12, 0)
        )

        tk.Label(
            status,
            textvariable=self.status_var,
            bg=BG,
            fg=MUTED,
            font=(FONT, 9),
        ).pack(side="left")

        tk.Label(
            status,
            text="Powered by Python · pandas · scikit-learn",
            bg=BG,
            fg=MUTED,
            font=(FONT, 9),
        ).pack(side="right")

    # ========================================================
    # STATISTICS CARD
    # ========================================================

    def _metric_card(
        self,
        parent,
        label,
        variable,
        column
    ):

        shadow = tk.Frame(
            parent,
            bg=SHADOW
        )

        shadow.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0 if column == 0 else 10, 0)
        )

        card = tk.Frame(
            shadow,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        card.pack(
            fill="both",
            expand=True,
            padx=(0, 2),
            pady=(0, 2)
        )

        tk.Label(
            card,
            text=label,
            bg=CARD,
            fg=MUTED,
            font=(FONT, 8, "bold"),
        ).pack(
            anchor="w",
            padx=14,
            pady=(11, 2)
        )

        tk.Label(
            card,
            textvariable=variable,
            bg=CARD,
            fg=TEXT,
            font=(FONT, 13, "bold"),
            wraplength=240,
        ).pack(
            anchor="w",
            padx=14,
            pady=(0, 11)
        )

        parent.grid_columnconfigure(
            column,
            weight=1,
            uniform="metric"
        )

    # ========================================================
    # BUTTONS WITH SHADOWS AND HOVER EFFECTS
    # ========================================================

    def _shadow_button(
        self,
        parent,
        text,
        command,
        primary=True
    ):

        shadow = tk.Frame(
            parent,
            bg=SHADOW
        )

        if primary:
            bg = BUTTON
            hover = BUTTON_HOVER
        else:
            bg = BUTTON_SECONDARY
            hover = BUTTON_SECONDARY_HOVER

        button = tk.Button(
            shadow,
            text=text,
            command=command,
            bg=bg,
            fg=TEXT,
            activebackground=hover,
            activeforeground=TEXT,
            font=(FONT, 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=13,
            pady=10,
            highlightthickness=0,
        )

        button.pack(
            padx=(0, 2),
            pady=(0, 2)
        )

        # Hover effect
        button.bind(
            "<Enter>",
            lambda _event: button.configure(
                bg=hover
            )
        )

        button.bind(
            "<Leave>",
            lambda _event: button.configure(
                bg=bg
            )
        )

        shadow.button = button

        return shadow

    # ========================================================
    # STATUS MESSAGE
    # ========================================================

    def _set_status(self, message):
        self.status_var.set(message)

    # ========================================================
    # SELECT DATASET
    # ========================================================

    def choose_dataset(self):

        path = filedialog.askopenfilename(
            title="Select course dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if path:
            self.load_dataset(path)

    # ========================================================
    # LOAD AND PROCESS DATASET
    # ========================================================

    def load_dataset(self, path):

        try:

            # Read CSV file
            df = pd.read_csv(path)

            # Required columns
            required = {
                "course_title",
                "subject"
            }

            missing = required.difference(
                df.columns
            )

            if missing:
                raise ValueError(
                    "This CSV is missing required column(s): "
                    + ", ".join(sorted(missing))
                )

            # Remove duplicate records
            df = df.drop_duplicates().copy()

            # Clean required columns
            df["course_title"] = (
                df["course_title"]
                .fillna("")
                .astype(str)
            )

            df["subject"] = (
                df["subject"]
                .fillna("")
                .astype(str)
            )

            if df.empty:
                raise ValueError(
                    "The selected CSV contains no course records."
                )

            # Add optional columns if missing
            for column in (
                "num_subscribers",
                "num_reviews"
            ):

                if column not in df.columns:
                    df[column] = 0

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(0)

            # Create text features for recommendations
            df["_feature_text"] = (
                df["course_title"].map(clean_text)
                + " "
                + df["subject"].map(clean_text)
            ).str.strip()

            # Convert course text into numerical vectors
            vectorizer = CountVectorizer(
                max_features=3000
            )

            vectors = vectorizer.fit_transform(
                df["_feature_text"]
            )

            # Calculate similarity between courses
            similarity = cosine_similarity(
                vectors
            )

            # Save processed data
            self.data = df.reset_index(
                drop=True
            )

            self.similarity = similarity

            self.course_titles = (
                self.data["course_title"].tolist()
            )

            self.csv_path = path

            # Update course dropdown
            self.search_box.configure(
                values=self.course_titles
            )

            if self.course_titles:
                self.course_var.set(
                    self.course_titles[0]
                )

            # Update statistics
            self.course_count_var.set(
                f"{len(self.data):,}"
            )

            self.subject_count_var.set(
                str(
                    self.data["subject"].nunique()
                )
            )

            self.dataset_var.set(
                Path(path).name
            )

            self._set_status(
                f"Loaded {len(self.data):,} courses successfully."
            )

            # Display popular courses
            self.show_popular()

        except Exception as exc:

            messagebox.showerror(
                "Could not load dataset",
                str(exc)
            )

            self._set_status(
                "Dataset load failed."
            )

    # ========================================================
    # FILTER COURSE DROPDOWN WHILE TYPING
    # ========================================================

    def _filter_course_choices(self, _event=None):

        if self.data.empty:
            return

        query = (
            self.course_var.get()
            .strip()
            .lower()
        )

        if not query:
            choices = self.course_titles

        else:
            choices = [
                title
                for title in self.course_titles
                if query in title.lower()
            ]

        # Limit suggestions for responsiveness
        self.search_box.configure(
            values=choices[:500]
        )

    # ========================================================
    # POPULARITY RECOMMENDATIONS
    # ========================================================

    def popularity_recommendations(
        self,
        top_n=5
    ):

        if self.data.empty:
            return self.data

        frame = self.data.copy()

        # Popularity score
        frame["_popularity_score"] = (
            0.6 * frame["num_subscribers"]
            + 0.4 * frame["num_reviews"]
        )

        return frame.sort_values(
            "_popularity_score",
            ascending=False
        ).head(top_n)

    # ========================================================
    # RECOMMEND SIMILAR COURSES
    # ========================================================

    def recommend(self):

        if (
            self.data.empty
            or self.similarity is None
        ):

            messagebox.showinfo(
                "Dataset required",
                "Load your udemy_courses.csv file first."
            )

            return

        title = self.course_var.get().strip()

        if not title:
            messagebox.showwarning(
                "Choose a course",
                "Please enter or select a course first."
            )
            return

        # Find exact course title
        matches = self.data.index[
            self.data["course_title"] == title
        ].tolist()

        # If exact match is unavailable, search partial titles
        if not matches:

            partial = self.data[
                self.data["course_title"].str.contains(
                    title,
                    case=False,
                    regex=False,
                    na=False
                )
            ]

            if len(partial) == 1:

                title = partial.iloc[0][
                    "course_title"
                ]

                self.course_var.set(title)

                matches = partial.index.tolist()

            elif len(partial) > 1:

                self._render_rows(
                    "Choose a matching course",
                    [
                        (
                            str(row["course_title"]),
                            None
                        )
                        for _, row in partial.head(10).iterrows()
                    ],
                    (
                        "Several titles matched. "
                        "Select one from the dropdown and try again."
                    )
                )

                return

            else:

                messagebox.showwarning(
                    "Course not found",
                    "Choose a course from the dropdown list."
                )

                return

        # Use the first matching course
        index = matches[0]

        # Retrieve similarity scores
        scores = self.similarity[index]

        # Sort courses by similarity
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )

        # Remove the selected course itself
        ranked = [
            (i, score)
            for i, score in ranked
            if i != index
        ][:6]

        # Prepare recommendation rows
        rows = [
            (
                self.data.iloc[i]["course_title"],
                f"{score * 100:.1f}% match"
            )
            for i, score in ranked
        ]

        # Display results
        self._render_rows(
            "Recommended for you",
            rows,
            (
                f"Courses most similar to “{title}” "
                "based on course title and subject."
            )
        )

        self._set_status(
            f"Showing {len(rows)} recommendations."
        )

    # ========================================================
    # DISPLAY POPULAR COURSES
    # ========================================================

    def show_popular(self):

        if self.data.empty:
            self._show_empty_state()
            return

        frame = self.popularity_recommendations(
            top_n=6
        )

        rows = [
            (
                row["course_title"],
                (
                    f'{row["_popularity_score"]:,.0f} '
                    "popularity score"
                )
            )
            for _, row in frame.iterrows()
        ]

        self._render_rows(
            "Popular courses",
            rows,
            (
                "Ranked by 60% subscribers + 40% reviews."
            )
        )

        self._set_status(
            "Showing the most popular courses."
        )

    # ========================================================
    # CLEAR RESULTS
    # ========================================================

    def _clear_results(self):

        for widget in self.results_body.winfo_children():
            widget.destroy()

    # ========================================================
    # RENDER RECOMMENDATION ROWS
    # ========================================================

    def _render_rows(
        self,
        heading,
        rows,
        note
    ):

        self.results_title.configure(
            text=heading
        )

        self.results_note.configure(
            text=note
        )

        self._clear_results()

        if not rows:
            self._show_empty_state()
            return

        for rank, (title, meta) in enumerate(
            rows,
            start=1
        ):

            # Individual result card
            row = tk.Frame(
                self.results_body,
                bg=CARD,
                highlightbackground=BORDER,
                highlightthickness=1
            )

            row.pack(
                fill="x",
                pady=5
            )

            # Rank badge colors
            if rank == 1:
                badge_bg = BUTTON
                badge_fg = TEXT
            else:
                badge_bg = BUTTON_SECONDARY
                badge_fg = ACCENT

            tk.Label(
                row,
                text=f"{rank:02d}",
                bg=badge_bg,
                fg=badge_fg,
                font=(FONT, 10, "bold"),
                width=4,
                pady=9,
            ).pack(
                side="left",
                padx=(10, 12),
                pady=7
            )

            # Course title
            tk.Label(
                row,
                text=str(title),
                bg=CARD,
                fg=TEXT,
                font=(FONT, 10, "bold"),
                anchor="w",
                justify="left",
                wraplength=600,
            ).pack(
                side="left",
                fill="x",
                expand=True,
                padx=(0, 12),
                pady=8
            )

            # Match percentage or popularity score
            if meta is not None:

                tk.Label(
                    row,
                    text=str(meta),
                    bg=CARD,
                    fg=GREEN,
                    font=(FONT, 9, "bold"),
                    anchor="e",
                ).pack(
                    side="right",
                    padx=12
                )

    # ========================================================
    # EMPTY STATE
    # ========================================================

    def _show_empty_state(self):

        self.results_title.configure(
            text="Your recommendations will appear here"
        )

        self.results_note.configure(
            text="Load a CSV dataset to get started"
        )

        self._clear_results()

        tk.Label(
            self.results_body,
            text=(
                "📚\n"
                "No course dataset loaded yet.\n\n"
                "Click “Load dataset” and select udemy_courses.csv."
            ),
            bg=CARD,
            fg=MUTED,
            font=(FONT, 12),
            justify="center",
        ).pack(
            fill="both",
            expand=True,
            pady=35
        )


# ============================================================
# RUN APPLICATION
# ============================================================

def main():

    root = tk.Tk()

    app = CourseRecommendationApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()