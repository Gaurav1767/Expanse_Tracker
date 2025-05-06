import Transaction from "../models/Transaction.js";

/**
 * Gets all transactions for the authenticated user.
 */
export const getTransactions = async (req, res) => {
  try {
    const userId = req.user.id;
    const transactions = await Transaction.find({ user: userId }).sort({ date: -1 });
    res.status(200).json(transactions);
  } catch (error) {
    console.error("Error fetching transactions:", error);
    res.status(500).json({ error: "Server error" });
  }
};

/**
 * Creates a new transaction for the authenticated user.
 */
export const createTransaction = async (req, res) => {
  try {
    const userId = req.user.id;
    const { description, category, type, amount, date } = req.body;
    
    // If transaction is an expense, verify that the user has sufficient balance
    if (type === "expense") {
      // Retrieve all transactions for the user
      const transactions = await Transaction.find({ user: userId });
      // Calculate current totals
      const totalIncome = transactions
        .filter(tx => tx.type === "income")
        .reduce((acc, tx) => acc + tx.amount, 0);
      const totalExpenses = transactions
        .filter(tx => tx.type === "expense")
        .reduce((acc, tx) => acc + tx.amount, 0);
      const balance = totalIncome - totalExpenses;
      
      if (balance < amount) {
        return res.status(400).json({ error: "Insufficient balance for this expense" });
      }
    }

    const newTransaction = new Transaction({
      user: userId,
      description,
      category,
      type,
      amount,
      date,
    });
    await newTransaction.save();
    res.status(201).json(newTransaction);
  } catch (error) {
    console.error("Error creating transaction:", error);
    res.status(500).json({ error: "Server error" });
  }
};

/**
 * Updates an existing transaction for the authenticated user.
 */
export const updateTransaction = async (req, res) => {
  try {
    const userId = req.user.id;
    const { id } = req.params;
    const updatedData = req.body;
    const transaction = await Transaction.findOneAndUpdate(
      { _id: id, user: userId },
      updatedData,
      { new: true }
    );
    if (!transaction) return res.status(404).json({ error: "Transaction not found" });
    res.status(200).json(transaction);
  } catch (error) {
    console.error("Error updating transaction:", error);
    res.status(500).json({ error: "Server error" });
  }
};

/**
 * Deletes a transaction for the authenticated user.
 */
export const deleteTransaction = async (req, res) => {
  try {
    const userId = req.user.id;
    const { id } = req.params;
    const transaction = await Transaction.findOneAndDelete({ _id: id, user: userId });
    if (!transaction) return res.status(404).json({ error: "Transaction not found" });
    res.status(200).json({ message: "Transaction deleted", transaction });
  } catch (error) {
    console.error("Error deleting transaction:", error);
    res.status(500).json({ error: "Server error" });
  }
};
